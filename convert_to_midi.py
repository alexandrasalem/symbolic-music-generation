from music21 import converter
from pathlib import Path
import os

orig_folder = 'new_simplified_bass_files_c/'
new_folder = 'new_simplified_bass_files_c_midi/'

os.makedirs(new_folder, exist_ok=True)

def convert_kern_to_midi(kern_file_path, midi_file_path):
    score = converter.parse(kern_file_path)
    score.write('midi', fp=midi_file_path)
    #print(f"Converted {kern_file_path} to {midi_file_path}")

files = list(Path(orig_folder).resolve().glob('*.krn'))

for file in files:
    new_filename = file.name.replace('.krn', '.mid')
    new_filepath = f'{new_folder}{new_filename}'
    try:
        convert_kern_to_midi(file, new_filepath)
    except Exception as e:
        # still need to look into the ones that don't work here
        print(file)
        print(e)
