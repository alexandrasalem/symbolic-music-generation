from torch.utils.data import Dataset
from pathlib import Path
import torch


class ChordMidiDataset(Dataset):
    def __init__(self, dataframe, midis_path, midi_tokenizer, chord_tokenizer, max_length=512):
        self.df = dataframe
        self.midi_tokenizer = midi_tokenizer
        self.chord_tokenizer = chord_tokenizer
        self.chord_max_length = max_length
        self.midi_paths = midis_path
        self.midi_max_length = max_length
        #self.pad_token = tuple(midi_tokenizer.pad_token_id for _ in range(len(midi_tokenizer.vocab)))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        tokenized = self.chord_tokenizer.encode(
            self.df.iloc[idx]['chord'],
        )
        input_ids, attn_mask = tokenized.ids, tokenized.attention_mask

        chord_pad_token = self.chord_tokenizer.token_to_id('[PAD]')
        if len(input_ids) < self.chord_max_length:
            pad_len = self.chord_max_length - len(input_ids)
            input_ids = input_ids + ([chord_pad_token] * pad_len)
            attn_mask = attn_mask + ([chord_pad_token] * pad_len)
        else:
            input_ids = input_ids[:self.chord_max_length]
            attn_mask = attn_mask[:self.chord_max_length]

        midi_id = self.df.iloc[idx]['name']
        midi_id = f'{midi_id}_score_simplified_melody_c' #f'{midi_id}_score_simplified_bass_c'
        midi_file_path = Path(self.midi_paths, f"{midi_id}.mid")
        midi_tokenized = self.midi_tokenizer(midi_file_path)
        midi_ids = midi_tokenized[0].ids  # List[List[int]] (T, F)
        midi_ids = [1] + midi_ids + [2] #salem trying this

        # Pad or truncate
        midi_tensor = torch.tensor(midi_ids, dtype=torch.long)
        T = midi_tensor.size(0)

        if T < self.midi_max_length:
            pad_len = self.midi_max_length - T
            pad_tensor = torch.full((pad_len,), self.midi_tokenizer.pad_token_id, dtype=torch.long)
            midi_tensor = torch.cat([midi_tensor, pad_tensor])
        else:
            midi_tensor = midi_tensor[:self.midi_max_length]
            midi_tensor[-1] = 2

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(attn_mask, dtype=torch.long), midi_tensor  # shapes: (L,), (L,), (T, F)