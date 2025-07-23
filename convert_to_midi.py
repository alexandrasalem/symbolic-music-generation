from music21 import converter
from pathlib import Path

def convert_kern_to_midi(kern_file_path, midi_file_path):
    score = converter.parse(kern_file_path)
    score.write('midi', fp=midi_file_path)
    print(f"Converted {kern_file_path} to {midi_file_path}")

files = list(Path('simplified_bass_files_c').resolve().glob('*.krn'))

for file in files:
    new_filename = file.name.replace('.krn', '.mid')
    new_filepath = 'simplified_bass_files_c_midi/' + new_filename
    try:
        convert_kern_to_midi(file, new_filepath)
    except Exception as e:
        print(file)
