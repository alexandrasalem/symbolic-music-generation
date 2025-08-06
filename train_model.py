import torch.optim as optim
import torch
import torch.nn as nn
from pathlib import Path
from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DatasetMIDI, DataCollator
from torch.utils.data import DataLoader
# from utils import load_pretrain_data, split_pretrain_data, compute_token_type_distribution
from models import RemiDecoder, ChordEncoder, Chord2MidiTransformer
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
import logging
import os
#from tqdm.auto import tqdm
import pandas as pd
from chord_to_midi_dataset import ChordMidiDataset
from tokenizers import Tokenizer

def validate(model, val_dataloader, criterion, device, epoch):
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)  # (B, T)
            attention_mask = batch['attention_mask'].to(device)

            decoder_input = input_ids[:, :-1]  # (B, T-1)
            tgt = input_ids[:, 1:]  # (B, T-1)
            attn_mask = attention_mask[:, :-1]
            tgt_key_padding_mask = (attn_mask == 0)

            logits = model(
                decoder_input,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory=None
            )  # (B, T-1, vocab_size)

            logits_flat = logits.reshape(-1, logits.size(-1))
            tgt_flat = tgt.reshape(-1)

            valid_tokens = (tgt_flat != criterion.ignore_index).sum().item()
            loss = criterion(logits_flat, tgt_flat)

            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

    avg_loss = total_loss / total_tokens
    log_msg = f"Epoch {epoch} - Validation Loss: {avg_loss:.4f}"
    print(log_msg)
    logging.info(log_msg)
    model.train()


def generate_samples(model, epoch, bos_id, eos_id, device, max_len=512, ):
    model.eval()
    os.makedirs(f"token_distribution/epoch_{epoch}", exist_ok=True)
    top_p_ids = model.generate(
        bos_id=bos_id,
        eos_id=eos_id,
        decoding_strategy="top_p",
        top_p=0.9,
        max_len=max_len,
        device=device
    )
    # token_distribution = compute_token_type_distribution(top_p_ids)
    # with open(f"token_distribution/epoch_{epoch}/sampled_track.pkl", 'wb') as f:
    #     pickle.dump(token_distribution, f)

    model.train()

bass_or_melody = "bass"

def main():
    logging.basicConfig(
        filename=f'chord2{bass_or_melody}_train_log.log',
        level=logging.INFO,
        format='%(asctime)s — %(levelname)s — %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    checkpoints_loc = f'chord2{bass_or_melody}_train_checkpoints'
    os.makedirs(checkpoints_loc, exist_ok=True)

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

    train_df = pd.read_csv("test_chords_edited.csv")
    midis_we_have = list(Path(f'new_simplified_{bass_or_melody}_files_c_midi').resolve().glob('*.mid'))
    midis_we_have = [item.name[:-22] for item in midis_we_have] #[item.name[:-28] for item in midis_we_have]
    train_df = train_df[train_df["long_name"].isin(midis_we_have)]
    midis_path = f"new_simplified_{bass_or_melody}_files_c_midi"
    train_dataset = ChordMidiDataset(
        train_df,
        midis_path=midis_path,
        midi_tokenizer=midi_tokenizer,
        chord_tokenizer=chord_tokenizer,
        bass_or_melody=bass_or_melody,
        max_length=128
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
            input_ids, attn_mask, tgt = [x.to(device) for x in batch]

            tgt_input = tgt[:, :-1]
            tgt_target = tgt[:, 1:]

            tgt_key_padding_mask = (tgt_input == midi_tokenizer.pad_token_id)

            logits = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
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
