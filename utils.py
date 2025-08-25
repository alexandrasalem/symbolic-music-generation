from miditok import REMI, TokenizerConfig, TokSequence
from miditok.utils import split_files_for_training
from pathlib import Path
import os
# import muspy
import music21
from random import shuffle
import torch
# import gdown
import zipfile
import pickle
from collections import defaultdict
from tqdm import tqdm
# import pretty_midi
# import pandas as pd


def convert_to_midi(token_ids, tokenizer, dump_path):
    """
    this function converts token_ids to actual tokens
    then the actual tokens are dumped into a midi file
    """
    id_to_token = {v: k for k, v in tokenizer.vocab.items()}
    actual_tokens = [[id_to_token[i] for i in token_ids]]
    
    with open("generated_tokens.txt", "w") as f:
        for token in actual_tokens[0]:
            f.write(token + "\n")
            
    generated_midi = tokenizer(actual_tokens)
    generated_midi.dump_midi(dump_path)
    

    

def generate_causal_mask(size):
    return torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)



def convert_to_midi_files(
    token_list,
    tokenizer,
    model_epoch,
    path
):
    
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

    token_strs = []

    for t in token_list:
        for k, v in tokenizer.vocab.items():
            if v == t:
                token_strs.append(k)
    
    seq = TokSequence(token_strs)
    score = tokenizer([seq])

    score.dump_midi(path)



def compute_token_type_distribution(midi_ids):
    counts = {
        "pitch": 0,
        "velocity": 0,
        "duration": 0,
        "position": 0,
        "bar": 0,
        "other": 0
    }

    for token_id in midi_ids:
        if 5 <= token_id <= 92:
            counts["pitch"] += 1
        elif token_id == 4:
            counts["bar"] += 1
        elif 93 <= token_id <= 124:
            counts["velocity"] += 1
        elif 125 <= token_id <= 188:
            counts["duration"] += 1
        elif 189 <= token_id <= 220:
            counts["position"] += 1
        else:
            counts["other"] += 1

    total = sum(counts.values())
    return {k: v / total if total > 0 else 0.0 for k, v in counts.items()}
