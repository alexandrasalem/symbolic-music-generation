from miditok import REMI, TokenizerConfig
from models import RemiDecoder, ChordEncoder, Chord2JointDecoderMidiTransformer
import torch
import os
from utils import convert_to_midi_files
from tokenizers import Tokenizer
import pandas as pd
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description="Arguments for controlling generation.")
parser.add_argument("--piece_or_theme", choices=["piece", "theme"], help="which train/test split")
parser.add_argument("--epoch", type=int, choices=[0, 50, 100, 150, 200, 250, 300, 350, 400], help="which epoch checkpoint")
args = parser.parse_args()

piece_or_theme = args.piece_or_theme
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
        checkpoint_loc = f"checkpoints/chord2jointdecoder_train_checkpoints/chord2jointdecoder_epoch_{epoch}.pt"
        samples_loc_stem = "samples/chord2jointdecoder_samples"
        my_chords_csv_path = "test_chords_edited-key-tranposed.csv"
        my_output_csv_path = "test_joint.csv"
    elif piece_or_theme == "theme":
        checkpoint_loc = f"checkpoints/chord2jointdecoder_theme_train_checkpoints/chord2jointdecoder_theme_epoch_{epoch}.pt"
        samples_loc_stem = "samples/chord2jointdecoder_theme_samples"
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
    bass_tokenizer = REMI(config)
    melody_tokenizer = REMI(config)
    chord_tokenizer = Tokenizer.from_file("chord_tokenizer.json")

    bos_id = 1
    eos_id = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = ChordEncoder(
        chord_tokenizer.get_vocab_size(),
        d_model=256,
        num_layers=1,
        nhead=2
    )

    decoder = RemiDecoder(
        len(bass_tokenizer.vocab),
        d_model=256,
        num_layers=6,
        nhead=8,
        include_linear_head=False
    )


    model = Chord2JointDecoderMidiTransformer(encoder = encoder, decoder = decoder, d_model = decoder.d_model, vocab_size=decoder.vocab_size)

    max_length = 128
    checkpoint = torch.load(checkpoint_loc, map_location=device)#torch.load(f"chord2jointdecoder_train_checkpoints/chord2jointdecoder_epoch_{epoch_to_load}.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)

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
    os.makedirs(f"{samples_loc_stem}/generated_midis_{epoch}/bass", exist_ok=True)
    os.makedirs(f"{samples_loc_stem}/generated_midis_{epoch}/melody", exist_ok=True)

    print("START GENERATING..")
    for idx, row in test_data.iterrows():
        tokenized = chord_tokenizer.encode(
            row['chord_transposed'],
        )
        input_ids, attn_mask = tokenized.ids, tokenized.attention_mask

        chord_pad_token = chord_tokenizer.token_to_id('[PAD]')
        if len(input_ids) < max_length:
            pad_len = max_length - len(input_ids)
            input_ids = input_ids + ([chord_pad_token] * pad_len)
            attn_mask = attn_mask + ([chord_pad_token] * pad_len)
        else:
            input_ids = input_ids[:max_length]
            attn_mask = attn_mask[:max_length]

        input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
        attn_mask = torch.tensor(attn_mask, dtype=torch.long).to(device)
        input_ids = input_ids.unsqueeze(0)
        attn_mask = attn_mask.unsqueeze(0)



        # generate samples
        bass_generated_ids, melody_generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            bos_id=bos_id,
            eos_id=eos_id,
            max_len=max_length,
            decoding_strategy="top_p",
            top_p=0.9,
            device=device
        )
        name = row['long_name']

        path = f"{samples_loc_stem}/generated_midis_{epoch}/bass/{name}_generated.mid"
        convert_to_midi_files(
            bass_generated_ids,
            bass_tokenizer,
            idx+1,
            path
        )

        path = f"{samples_loc_stem}/generated_midis_{epoch}/melody/{name}_generated.mid"
        convert_to_midi_files(
            melody_generated_ids,
            melody_tokenizer,
            idx+1,
            path
        )


if __name__ == "__main__":
    main()