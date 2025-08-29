import torch.optim as optim
import torch
import torch.nn as nn
from pathlib import Path
from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DatasetMIDI, DataCollator
from torch.utils.data import DataLoader
# from utils import load_pretrain_data, split_pretrain_data, compute_token_type_distribution
from models import RemiDecoder, SequentialRemiDecoder, ChordEncoder, Chord2SequentialMidiTransformer
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
import logging
import os
#from tqdm.auto import tqdm
import pandas as pd
from chord_to_midi_dataset import ChordBassMelodyDataset, ChordMidiDataset
from tokenizers import Tokenizer

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
    if bass_first:
        bass_or_melody = 'bass_first'
    else:
        bass_or_melody = 'melody_first'
    logging.basicConfig(
        filename=f'chord2sequential{bass_or_melody}_train_log.log',
        level=logging.INFO,
        format='%(asctime)s — %(levelname)s — %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    checkpoints_loc = f'chord2sequential{bass_or_melody}_train_checkpoints'
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
    bass_tokenizer = midi_tokenizer
    melody_tokenizer = midi_tokenizer
    chord_tokenizer = Tokenizer.from_file("chord_tokenizer.json")

    train_df = construct_train_df(
        chords_csv_path="train_chords_edited.csv",
        bass_folder="new_simplified_bass_files_c_midi",
        melody_folder="new_simplified_melody_files_c_midi",
        output_csv_path="train_joint.csv"
    )

    train_dataset = ChordBassMelodyDataset(
        dataframe=train_df,
        chord_tokenizer=chord_tokenizer,
        bass_tokenizer=bass_tokenizer,
        melody_tokenizer=melody_tokenizer,
        max_length=128,
    )
    batch_size = 8
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

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
            bass_input_ids = batch["bass_input_ids"].to(device)
            melody_input_ids = batch["melody_input_ids"].to(device)

            # Prepare targets for bass and melody
            bass_input = bass_input_ids[:, :-1]
            bass_target = bass_input_ids[:, 1:]
            melody_input = melody_input_ids[:, :-1]
            melody_target = melody_input_ids[:, 1:]



            if bass_first:
                first_tgt_key_padding_mask = (bass_input == bass_tokenizer.pad_token_id)
                second_tgt_key_padding_mask = (melody_input == melody_tokenizer.pad_token_id)
                bass_logits, melody_logits = model(
                    chord_input_ids = chord_input_ids,
                    chord_attention_mask = chord_attention_mask,
                    first_input_ids = bass_input,
                    second_input_ids = melody_input,
                    first_tgt_key_padding_mask=first_tgt_key_padding_mask,
                    second_tgt_key_padding_mask=second_tgt_key_padding_mask,
                )
            else:
                first_tgt_key_padding_mask = (melody_input == melody_tokenizer.pad_token_id)
                second_tgt_key_padding_mask = (bass_input == bass_tokenizer.pad_token_id)
                melody_logits, bass_logits = model(
                    chord_input_ids = chord_input_ids,
                    chord_attention_mask = chord_attention_mask,
                    first_input_ids = melody_input,
                    second_input_ids = bass_input,
                    first_tgt_key_padding_mask=first_tgt_key_padding_mask,
                    second_tgt_key_padding_mask=second_tgt_key_padding_mask,
                )

            bass_loss = criterion(bass_logits.reshape(-1, bass_logits.size(-1)), bass_target.reshape(-1))
            melody_loss = criterion(melody_logits.reshape(-1, melody_logits.size(-1)), melody_target.reshape(-1))
            loss = bass_loss + melody_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
        print(f"Epoch {epoch} — Loss: {epoch_loss / len(train_dataloader) * batch_size:.4f}")

        if epoch % save_every == 0:
            checkpoint_path = f"{checkpoints_loc}/chord2sequential{bass_or_melody}_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

if __name__ == "__main__":
    bass_first = True
    main()
