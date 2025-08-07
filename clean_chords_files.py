import re
import mir_eval

def update_chords(chords_string, keep_na):
    replacement_chords = {'G#': 'Ab', 'A#': 'Bb', 'B#': 'Cb', 'C#': 'Db', 'D#': 'Eb', 'E#': 'Fb', 'F#': 'Gb'}
    chords = chords_string.split(" ")
    new_chords = []
    for chord in chords:
        if 'NA' in chord:
            print(chord)
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
    new_chords_joined = update_chords(elts[2], keep_na=True)
    new_line = f'{elts[0]},{elts[1]},{new_chords_joined}'
    new_lines.append(new_line)

new_lines = ['name,long_name,chord\n'] + new_lines

with open('test_chords_edited.csv', 'w') as f:
    for chord in new_lines:
        f.write(chord)