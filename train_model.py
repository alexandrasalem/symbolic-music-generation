import torch.optim as optim
import torch
import torch.nn as nn
from pathlib import Path
from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DatasetMIDI, DataCollator
from torch.utils.data import DataLoader
from models import RemiDecoder, ChordEncoder, Chord2MidiTransformer
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
import logging
import os
#from tqdm.auto import tqdm
import pandas as pd
from chord_to_midi_dataset import ChordBassMelodyDataset, ChordMidiDataset
from tokenizers import Tokenizer
import argparse

parser = argparse.ArgumentParser(description="Arguments for controlling training independent.")
parser.add_argument("bass_or_melody", default="bass", help="which voice to run")
parser.add_argument("piece_or_theme", default="piece", help="which train/test split")
args = parser.parse_args()

bass_or_melody = args.bass_or_melody
piece_or_theme = args.piece_or_theme

def extract_prefix(filename):
    # Remove extension and everything after '_simplified'
    stem = Path(filename).stem
    return stem.split('_simplified')[0]

def construct_train_df(
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
        logs_filesname = f'chord2{bass_or_melody}_train_log.log'
        my_chords_csv_path = "train_chords_edited.csv"
        my_output_csv_path = "train_joint.csv"
        checkpoints_loc = f'chord2{bass_or_melody}_train_checkpoints'
    elif piece_or_theme == "theme":
        logs_filesname = f'chord2{bass_or_melody}_theme_train_log.log'
        my_chords_csv_path = "train_themes_held_out_chords_edited.csv"
        my_output_csv_path = "train_joint_themes_held_out.csv"
        checkpoints_loc = f'chord2{bass_or_melody}_theme_train_checkpoints'
    else:
        raise ValueError(f"Unknown piece or theme type: {piece_or_theme}")

    logging.basicConfig(
        filename=logs_filesname,
        level=logging.INFO,
        format='%(asctime)s — %(levelname)s — %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


    os.makedirs(checkpoints_loc, exist_ok=True)

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
    chord_tokenizer = Tokenizer.from_file("chord_tokenizer.json")


    train_df = construct_train_df(
        chords_csv_path=my_chords_csv_path,
        bass_folder="new_simplified_bass_files_c_midi",
        melody_folder="new_simplified_melody_files_c_midi",
        output_csv_path=my_output_csv_path
    )


    train_dataset = ChordBassMelodyDataset(
        dataframe=train_df,
        chord_tokenizer=chord_tokenizer,
        bass_tokenizer=midi_tokenizer,
        melody_tokenizer=midi_tokenizer,
        max_length=128,
    )

    batch_size = 8
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)


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

    # use multiple gpu if available
    # if torch.cuda.device_count() > 1:
    #    print(f"Using {torch.cuda.device_count()} GPUs")
    #    model = nn.DataParallel(model)

    model.to(device)
    print(f"model moved to {device}!")

    warmup_steps = 1000

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0

    criterion = nn.CrossEntropyLoss(ignore_index=midi_tokenizer.pad_token_id)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = LambdaLR(optimizer, lr_lambda)

    # load from checkpoints
    # model, optimizer, start_epoch = load_checkpoint(model, optimizer, "pretrain_checkpoints")
    # print(f"Loaded checkpoints! starting training from EPOCH: {start_epoch}: ")

    start_epoch = 0
    num_epochs = 401
    save_every = 50
    val_every = 50
    log_interval = 1000

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            chord_input_ids = batch["chord_input_ids"].to(device)
            chord_attention_mask = batch["chord_attention_mask"].to(device)
            midi_input_ids = batch[f"{bass_or_melody}_input_ids"].to(device)

            # Prepare targets for bass and melody
            tgt_input = midi_input_ids[:, :-1]
            tgt_target = midi_input_ids[:, 1:]

            tgt_key_padding_mask = (tgt_input == midi_tokenizer.pad_token_id)

            logits = model(
                input_ids=chord_input_ids,
                attention_mask=chord_attention_mask,
                tgt=tgt_input,
                tgt_key_padding_mask=tgt_key_padding_mask
            )

            logits_flat = logits.reshape(-1, logits.size(-1))
            tgt_target_flat = tgt_target.reshape(-1)
            loss = criterion(logits_flat, tgt_target_flat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
        log_msg = f"Epoch {epoch} — Loss: {epoch_loss / len(train_dataloader) * batch_size:.4f}"
        print(log_msg)
        logging.info(log_msg)

        if epoch % save_every == 0:# and epoch != 0:
            checkpoint_path = f"{checkpoints_loc}/chord2{bass_or_melody}_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if isinstance(model,
                                                                            nn.DataParallel) else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

        # avg_loss = epoch_loss / len(train_dataloader)
        # logging.info(f"Epoch {epoch} — Loss: {avg_loss:.4f}")
        # print(f"Epoch {epoch} — Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    main()
