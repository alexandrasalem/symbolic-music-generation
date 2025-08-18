from torch.utils.data import Dataset
from pathlib import Path
import torch


class ChordMidiDataset(Dataset):
    def __init__(self, dataframe, midis_path, midi_tokenizer, chord_tokenizer, bass_or_melody, max_length=512):
        self.df = dataframe
        self.midi_tokenizer = midi_tokenizer
        self.chord_tokenizer = chord_tokenizer
        self.chord_max_length = max_length
        self.midi_paths = midis_path
        self.bass_or_melody = bass_or_melody
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

        midi_id = self.df.iloc[idx]['long_name']
        midi_id = f'{midi_id}_simplified_{self.bass_or_melody}_c' #f'{midi_id}_score_simplified_bass_c'
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

class ChordBassMelodyDataset(Dataset):
    def __init__(
        self,
        dataframe,
        chord_tokenizer,
        bass_tokenizer,
        melody_tokenizer,
        max_length=128,
    ):
        self.df = dataframe
        self.chord_tokenizer = chord_tokenizer
        self.bass_tokenizer = bass_tokenizer
        self.melody_tokenizer = melody_tokenizer
        self.chord_max_length = max_length
        self.bass_max_length = max_length
        self.melody_max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Chord
        chord_tokenized = self.chord_tokenizer.encode(row['chord'])
        chord_input_ids, chord_attn_mask = chord_tokenized.ids, chord_tokenized.attention_mask
        chord_pad_token = self.chord_tokenizer.token_to_id('[PAD]')
        if len(chord_input_ids) < self.chord_max_length:
            pad_len = self.chord_max_length - len(chord_input_ids)
            chord_input_ids += [chord_pad_token] * pad_len
            chord_attn_mask += [chord_pad_token] * pad_len
        else:
            chord_input_ids = chord_input_ids[:self.chord_max_length]
            chord_attn_mask = chord_attn_mask[:self.chord_max_length]

        # Bass
        bass_file_path = row['bass_path']
        bass_tokenized = self.bass_tokenizer(bass_file_path)
        # print bass tokenized for debugging
        # print(bass_tokenized)
        bass_ids = bass_tokenized[0].ids
        bass_ids = [1] + bass_ids + [2]
        bass_pad_token = self.bass_tokenizer.pad_token_id
        bass_tensor = torch.tensor(bass_ids, dtype=torch.long)
        T_bass = bass_tensor.size(0)

        if T_bass < self.bass_max_length:
            pad_len = self.bass_max_length - T_bass
            pad_tensor = torch.full((pad_len,), bass_pad_token, dtype=torch.long)
            bass_tensor = torch.cat([bass_tensor, pad_tensor])
        else:
            bass_tensor = bass_tensor[:self.bass_max_length]
            bass_tensor[-1] = 2 #self.bass_tokenizer.token_to_id('[EOS]')

        # Melody
        melody_file_path = row['melody_path']
        melody_tokenized = self.melody_tokenizer(melody_file_path)
        melody_ids = melody_tokenized[0].ids
        melody_ids = [1] + melody_ids + [2]
        melody_pad_token = self.melody_tokenizer.pad_token_id
        melody_tensor = torch.tensor(melody_ids, dtype=torch.long)
        T_melody = melody_tensor.size(0)
        if T_melody < self.melody_max_length:
            pad_len = self.melody_max_length - T_melody
            pad_tensor = torch.full((pad_len,), melody_pad_token, dtype=torch.long)
            melody_tensor = torch.cat([melody_tensor, pad_tensor])
        else:
            melody_tensor = melody_tensor[:self.melody_max_length]
            melody_tensor[-1] = 2 #self.melody_tokenizer.token_to_id('[EOS]') # ignoring this for now
            # TODO: add EOS token if needed

        return {
            "chord_input_ids": torch.tensor(chord_input_ids, dtype=torch.long),
            "chord_attention_mask": torch.tensor(chord_attn_mask, dtype=torch.long),
            "bass_input_ids": bass_tensor,
            "melody_input_ids": melody_tensor,
        }