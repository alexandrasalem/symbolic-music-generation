from tokenizers import Tokenizer, processors, decoders
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import WhitespaceSplit
import pandas as pd
import re
import mir_eval

def update_chords(chords_string, keep_na):
    replacement_chords = {'G#': 'Ab', 'A#': 'Bb', 'B#': 'Cb', 'C#': 'Db', 'D#': 'Eb', 'E#': 'Fb', 'F#': 'Gb'}
    chords = chords_string.split(" ")
    new_chords = []
    for chord in chords:
        if 'NA' in chord:
            if "/" in chord and re.split("/", chord)[1] == "NA":
                chord = re.split("/", chord)[0]
            else:
                if keep_na:
                    new_chords.append("NA")
                    continue
                else:
                    continue
        chord_edited = chord.replace("-", "b")
        reduced_chord_split = mir_eval.chord.split(chord_edited)
        second_half = reduced_chord_split[1]
        second_half = re.sub(r"\d", "", second_half)
        second_half = re.sub("dim", "min", second_half)
        second_half = re.sub("aug", "maj", second_half)
        if len(second_half) == 0:
            second_half = "maj"
        if second_half[0] == "h":
            second_half = second_half[1:]
        new_chord = f'{reduced_chord_split[0]}:{second_half}'
        for key, value in replacement_chords.items():
            new_chord = new_chord.replace(key, value)
        new_chords.append(new_chord)
    new_chords_joined = " ".join(new_chords)
    new_chords_joined = new_chords_joined + "\n"
    return new_chords_joined

with open('test_chords.txt') as f:
    lines = f.readlines()

lines = [line.replace("\n", "") for line in lines]
new_lines = []
for line in lines:
    new_chords_joined = update_chords(line, keep_na=False)
    new_lines.append(new_chords_joined)

with open('test_chords_edited.txt', 'w') as f:
    for chord in new_lines:
        f.write(chord)

with open('test_chords.tsv') as f:
    all_lines = f.readlines()

lines = [line.replace("\n", "") for line in all_lines[1:]]
new_lines = []
for line in lines:
    elts = line.split("\t")
    new_chords_joined = update_chords(elts[1], keep_na=True)
    new_line = f'{elts[0]},{new_chords_joined}'
    new_lines.append(new_line)

new_lines = ['name,chord\n'] + new_lines

with open('test_chords_edited.csv', 'w') as f:
    for chord in new_lines:
        f.write(chord)


# Initialize an empty tokenizer
tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
tokenizer.pre_tokenizer = WhitespaceSplit()

print(tokenizer.pre_tokenizer.pre_tokenize_str("C:min G:maj/3 G:7/3 C:min"))
# Trainer with special tokens
trainer = WordLevelTrainer(
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    min_frequency=1,
)

# Train from file
tokenizer.train(["test_chords_edited.txt"], trainer)

encoding = tokenizer.encode("C:min G:maj G:maj G:maj C:min")
print(encoding.tokens)

cls_token_id = tokenizer.token_to_id("[CLS]")
sep_token_id = tokenizer.token_to_id("[SEP]")
print(cls_token_id, sep_token_id)

tokenizer.post_processor = processors.TemplateProcessing(
    single=f"[CLS]:0 $A:0 [SEP]:0",
    pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
    special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)],
)

tokenizer.decoder = decoders.WordPiece(prefix='##')


# saving vocab
sorted_items = dict(sorted(tokenizer.get_vocab().items()))  # returns a list of tuples
df = pd.DataFrame.from_dict(sorted_items, orient='index', columns=['count'])
df['chord'] = df.index
df = df.loc[:, ['chord']]
df.to_csv("chords.csv", index=False)

tokenizer.save("test_chord_tokenizer.json")

chord_tokenizer = Tokenizer.from_file("test_chord_tokenizer.json")
print(chord_tokenizer.encode('C:min NA G:maj G:maj C:min').tokens)