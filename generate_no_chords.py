from miditok import REMI, TokenizerConfig
from models import RemiDecoder
import torch
import os
from tqdm import tqdm
from utils import convert_to_midi_files
import pandas as pd
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description="Arguments for controlling generation.")
parser.add_argument("--piece_or_theme", choices=["piece", "theme"], help="which train/test split")
parser.add_argument("--bass_or_melody", choices=["bass", "melody"], help="which voice to run")
parser.add_argument("--epoch", type=int, choices=[0, 50, 100, 150, 200, 250, 300, 350, 400], help="which epoch checkpoint")
args = parser.parse_args()

piece_or_theme = args.piece_or_theme
bass_or_melody = args.bass_or_melody
epoch = args.epoch
def extract_prefix(filename):
    # Remove extension and everything after '_simplified'
    stem = Path(filename).stem
    return stem.split('_simplified')[0]

def construct_test_df(
    chords_csv_path,
    melody_folder,
    bass_folder,
    output_csv_path=None
):
    df = pd.read_csv(chords_csv_path)
    melody_files = {extract_prefix(f): str(f.resolve()) for f in Path(melody_folder).glob("*.mid")}
    bass_files = {extract_prefix(f): str(f.resolve()) for f in Path(bass_folder).glob("*.mid")}

    df["melody_path"] = df["long_name"].map(melody_files)
    df["bass_path"] = df["long_name"].map(bass_files)

    df = df.dropna(subset=["melody_path", "bass_path"])

    if output_csv_path:
        df.to_csv(output_csv_path, index=False)

    print(f"Data Length: {len(df)}")
    return df

def main():
    # set based on argparse
    if piece_or_theme == "piece":
        checkpoint_loc = f"checkpoints/nothingprior2{bass_or_melody}_train_checkpoints/nothingprior2{bass_or_melody}_epoch_{epoch}.pt"
        samples_loc_stem = f"samples/nothingprior2{bass_or_melody}_samples"
        my_chords_csv_path = "test_chords_edited-key-tranposed.csv"
        my_output_csv_path = "test_joint.csv"
    elif piece_or_theme == "theme":
        checkpoint_loc = f"checkpoints/nothingprior2{bass_or_melody}_theme_train_checkpoints/nothingprior2{bass_or_melody}_theme_epoch_{epoch}.pt"
        samples_loc_stem = f"samples/nothingprior2{bass_or_melody}_theme_samples"
        my_chords_csv_path = "test_themes_held_out_chords_edited-key-tranposed.csv"
        my_output_csv_path = "test_joint_themes_held_out.csv"
    else:
        raise ValueError(f"Unknown piece or theme type: {piece_or_theme}")

    TOKENIZER_PARAMS = {
    "pitch_range": (21, 109),
    "beat_res": {(0, 4): 8, (4, 12): 4},
    "num_velocities": 32,
    "special_tokens": ["PAD", "BOS", "EOS", "MASK"],
    "use_chords": False,
    "use_rests": True,
    "use_tempos": True,
    "use_time_signatures": False,
    "use_programs": False,
    "num_tempos": 32,  # number of tempo bins
    "tempo_range": (40, 250),  # (min, max)
    }
    config = TokenizerConfig(**TOKENIZER_PARAMS)
    tokenizer = REMI(config)

    bos_id = 1
    eos_id = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RemiDecoder(
        len(tokenizer.vocab),
        d_model=256,
        num_layers=6,
        nhead=8
    ).to(device)

    checkpoint = torch.load(checkpoint_loc, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Model loaded from checkpoint!")

    test_data = construct_test_df(
        chords_csv_path=my_chords_csv_path,
        bass_folder="new_simplified_bass_files_c_midi",
        melody_folder="new_simplified_melody_files_c_midi",
        output_csv_path=my_output_csv_path
    )
    print("Test data (chord progressions) loaded!")

    os.makedirs(f"{samples_loc_stem}", exist_ok=True)
    os.makedirs(f"{samples_loc_stem}/generated_midis_{epoch}", exist_ok=True)
    print("START GENERATING..")
    for i in tqdm(range(len(test_data))):
        top_p_ids = model.generate(
            bos_id=bos_id,
            eos_id=eos_id,
            decoding_strategy="top_p",
            top_p=0.9,
            max_len=128,
            device=device
        )

        path = f"{samples_loc_stem}/generated_midis_{epoch}/track_{i+1}.mid"
        convert_to_midi_files(
            top_p_ids,
            tokenizer,
            i+1,
            path
        )

if __name__ == "__main__":
    main()