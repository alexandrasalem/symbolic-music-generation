from miditok import REMI, TokenizerConfig
from models import RemiDecoder, SequentialRemiDecoder, ChordEncoder, Chord2SequentialMidiTransformer
import torch
import os
from utils import convert_to_midi_files
from tokenizers import Tokenizer
import pandas as pd
from pathlib import Path

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
    if bass_first:
        bass_or_melody = 'bass_first'
    else:
        bass_or_melody = 'melody_first'
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
    midi_tokenizer = REMI(config)
    bass_tokenizer = midi_tokenizer
    melody_tokenizer = midi_tokenizer
    chord_tokenizer = Tokenizer.from_file("chord_tokenizer.json")

    bos_id = 1
    eos_id = 2

    chord_encoder = ChordEncoder(
        vocab_size=chord_tokenizer.get_vocab_size(),
        d_model=256,
        num_layers=2,
        nhead=4
    )
    first_decoder = RemiDecoder(
        vocab_size=len(midi_tokenizer.vocab),
        d_model=256,
        num_layers=2,
        nhead=4
    )
    second_decoder = SequentialRemiDecoder(
        vocab_size=len(midi_tokenizer.vocab),
        d_model=256,
        num_layers=4,
        nhead=8
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Chord2SequentialMidiTransformer(chord_encoder, first_decoder, second_decoder)
    model.to(device)

    max_length = 128
    checkpoint = torch.load(f"chord2sequential{bass_or_melody}_train_checkpoints/chord2sequential{bass_or_melody}_epoch_{epoch_to_load}.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)

    print("Model loaded from checkpoint!")

    test_data = construct_test_df(
        chords_csv_path="test_chords_edited.csv",
        bass_folder="new_simplified_bass_files_c_midi",
        melody_folder="new_simplified_melody_files_c_midi",
        output_csv_path="test_joint.csv"
    )
    print("Test data (chord progressions) loaded!")

    #os.makedirs(f"token_distribution/epoch_{epoch_to_load}", exist_ok=True)
    os.makedirs(f"chord2sequential{bass_or_melody}_samples", exist_ok=True)
    os.makedirs(f"chord2sequential{bass_or_melody}_samples/generated_midis_{epoch_to_load}", exist_ok=True)
    os.makedirs(f"chord2sequential{bass_or_melody}_samples/generated_midis_{epoch_to_load}/bass", exist_ok=True)
    os.makedirs(f"chord2sequential{bass_or_melody}_samples/generated_midis_{epoch_to_load}/melody", exist_ok=True)

    print("START GENERATING..")
    for idx, row in test_data.iterrows():
        tokenized = chord_tokenizer.encode(
            row['chord'],
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
        if bass_first:
            bass_generated_ids, melody_generated_ids = model.generate(
                chord_input_ids=input_ids,
                chord_attention_mask=attn_mask,
                first_bos_id=bos_id,
                first_eos_id=eos_id,
                second_bos_id=bos_id,
                second_eos_id=eos_id,
                max_len=max_length,
                decoding_strategy="top_p",
                top_p=0.9,
                device=device
            )
        else:
            melody_generated_ids, bass_generated_ids = model.generate(
                chord_input_ids=input_ids,
                chord_attention_mask=attn_mask,
                first_bos_id=bos_id,
                first_eos_id=eos_id,
                second_bos_id=bos_id,
                second_eos_id=eos_id,
                max_len=max_length,
                decoding_strategy="top_p",
                top_p=0.9,
                device=device
            )
        name = row['long_name']

        path = f"chord2sequential{bass_or_melody}_samples/generated_midis_{epoch_to_load}/bass/{name}_generated.mid"
        convert_to_midi_files(
            bass_generated_ids,
            bass_tokenizer,
            idx+1,
            path
        )

        path = f"chord2sequential{bass_or_melody}_samples/generated_midis_{epoch_to_load}/melody/{name}_generated.mid"
        convert_to_midi_files(
            melody_generated_ids,
            melody_tokenizer,
            idx+1,
            path
        )


if __name__ == "__main__":
    test_data_loc = "test_chords_edited.csv"
    bass_first = False
    epoch_to_load = 400
    main()