import music21
from music21 import converter
from pathlib import Path
import os
import pandas as pd
import partitura as pt

music21.defaults.ticksAtStart = 0

def convert_kern_to_midi(kern_file_path, midi_file_path):
    score = converter.parse(kern_file_path)
    score.write('midi', fp=midi_file_path)
    # score = pt.load_kern(kern_file_path)
    # pt.save_score_midi(score, midi_file_path)

def process_kern_folder(orig_kern_folder_path, new_midi_folder_path):
    os.makedirs(new_midi_folder_path, exist_ok=True)
    files = list(Path(orig_kern_folder_path).resolve().glob('*.krn'))

    problem_files = []
    problem_errors = []

    for file in files:
        new_filename = file.name.replace('.krn', '.mid')
        new_filepath = f'{new_midi_folder_path}{new_filename}'
        try:
            convert_kern_to_midi(file, new_filepath)
        except Exception as e:
            # still need to look into the ones that don't work here
            problem_files.append(file.name)
            problem_errors.append(e)
            print("problem file", file.name)

    #problem_df = pd.DataFrame({'problem_files': problem_files, 'problem_file_errors': problem_errors})
    #problem_df.to_csv('problem_files_converting_to_midi_melody.csv', index=False)

#orig_folder = 'new_new_simplified_bass_files_c/'
#new_folder = 'new_simplified_bass_files_c_midi/'

# orig_folder = 'new_simplified_bass_files_c/'
# new_folder = 'new_simplified_bass_files_c_midi/'

orig_folder = 'new_simplified_melody_files_c/'
new_folder = 'new_simplified_melody_files_c_midi/'

process_kern_folder(orig_folder, new_folder)

