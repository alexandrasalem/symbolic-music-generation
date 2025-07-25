import argparse

from miditok import REMI, TokenizerConfig
from models import RemiDecoder, ChordEncoder, Chord2MidiTransformer
import torch
import os
from tqdm import tqdm
from utils import convert_to_midi_files, compute_token_type_distribution
from tokenizers import Tokenizer
import pandas as pd

def main():
    TOKENIZER_PARAMS = {
    "pitch_range": (21, 109),
    "beat_res": {(0, 4): 8, (4, 12): 4},
    "num_velocities": 32,
    "special_tokens": ["PAD", "BOS", "EOS", "MASK"],
    "use_chords": False,
    "use_rests": False,
    "use_tempos": True,
    "use_time_signatures": False,
    "use_programs": False,
    "num_tempos": 32,  # number of tempo bins
    "tempo_range": (40, 250),  # (min, max)
    }
    config = TokenizerConfig(**TOKENIZER_PARAMS)
    midi_tokenizer = REMI(config)
    chord_tokenizer = Tokenizer.from_file("test_chord_tokenizer.json")

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
        len(midi_tokenizer.vocab),
        d_model=256,
        num_layers=6,
        nhead=8
    )
    model = Chord2MidiTransformer(encoder, decoder)

    epoch_to_load = 400
    bass_or_melody = "bass"
    checkpoint = torch.load(f"train_checkpoints/{bass_or_melody}/chord2midi_epoch_{epoch_to_load}.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Model loaded from checkpoint!")

    test_data = pd.read_csv("test_chords_edited.csv")
    test_data = test_data[:15]
    print("Test data (chord progressions) loaded!")

    #os.makedirs(f"token_distribution/epoch_{epoch_to_load}", exist_ok=True)
    os.makedirs("chord2midi_samples", exist_ok=True)
    os.makedirs(f"chord2midi_samples/{bass_or_melody}/generated_midis_{epoch_to_load}", exist_ok=True)

    print("START GENERATING..")
    for idx, row in test_data.iterrows():
        chord_text = row["chord"]
        inputs = chord_tokenizer.encode(chord_text)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # generate samples
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bos_id=bos_id,
            eos_id=eos_id,
            max_len=256,
            decoding_strategy="top_p",
            top_p=0.9,
            device=device
        )
        path = f"chord2midi_samples/{bass_or_melody}/generated_midis_{epoch_to_load}/track_{idx+1}.mid"
        convert_to_midi_files(
            generated_ids,
            midi_tokenizer,
            idx+1,
            path
        )


if __name__ == "__main__":
    main()