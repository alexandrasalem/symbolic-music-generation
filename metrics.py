from music21 import converter, note, chord, stream, interval, analysis, pitch
import pandas as pd
from pathlib import Path
import os
import math

from music21.chord import Chord

sharp_to_flat_note_dict = {
    "C#": "D-",
    "D#": "E-",
    "E#": "F-",
    "F#": "G-",
    "G#": "A-",
    "A#": "B-",
    "B#": "C-",
}

flat_to_sharp_note_dict = {value: key for key, value in sharp_to_flat_note_dict.items()}

chord_to_note_dict = {
    "C:maj": ["C", "E", "G"],
    "C:min": ["C", "E-", "G"],
    "Db:maj": ["D-", "F", "A-"],
    "Db:min": ["D-", "E", "A-"],
    "D:maj": ["D", "F#", "A"],
    "D:min": ["D", "F", "A"],
    "Eb:maj": ["E-", "G", "B-"],
    "Eb:min": ["E-", "G-", "B-"],
    "E:maj": ["E", "G#", "B"],
    "E:min": ["E", "G", "B"],
    "F:maj": ["F", "A", "C"],
    "F:min": ["F", "A-", "C"],
    "Gb:maj": ["G-", "B-", "D-"],
    "Gb:min": ["G-", "A", "D-"],
    "G:maj": ["G", "B", "D"],
    "G:min": ["G", "B-", "D"],
    "Ab:maj": ["A-", "C", "E-"],
    "Ab:min": ["A-", "B", "E-"],
    "A:maj": ["A", "C#", "E"],
    "A:min": ["A", "C", "E"],
    "Bb:maj": ["B-", "D", "F"],
    "Bb:min": ["B-", "D-", "F"],
    "B:maj": ["B", "D#", "F#"],
    "B:min": ["B", "D", "F#"],
}

def add_to_chord_to_note_dict():
    for key, value in chord_to_note_dict.items():
        for item in value.copy():
            if "#" in item:
                chord_to_note_dict[key].append(sharp_to_flat_note_dict[item])
            elif "-" in item:
                chord_to_note_dict[key].append(flat_to_sharp_note_dict[item])
            else:
                continue
    print("done")

def note_present_in_chord(note, chord):
    if chord in chord_to_note_dict.keys():
        if note in chord_to_note_dict[chord]:
            return True
    else:
        print(f"{chord} not in dictionary")
    return False

def grab_notes(midi_file):
    flat_notes = midi_file.flatten()#.notes
    note_list = []

    for element in flat_notes:
        if isinstance(element, note.Note):
            note_list.append(str(element.name))
        elif isinstance(element, note.Rest):
            note_list.append("r")
        elif isinstance(element, chord.Chord):
            note_list.append("c")
        else:
            continue
    return note_list

def grab_notes_trimmed(midi_file, chords_for_file, with_octave=False):
    flat_notes = midi_file.flatten()#.notes
    note_list = []
    for element in flat_notes:
        if isinstance(element, note.Note):
            if not with_octave:
                note_list.append(str(element.name))
            else:
                note_list.append(str(element.nameWithOctave))
        elif isinstance(element, note.Rest):
            note_list.append("r")
        elif isinstance(element, chord.Chord):
            note_list.append("c")
        else:
            continue

    if len(chords_for_file) != len(note_list):
        if len(chords_for_file) > len(note_list):
            new_note_list = note_list.copy() + ["NA"] * (len(chords_for_file) - len(note_list))  # pad generation
        else:
            new_note_list = note_list.copy()[:len(chords_for_file)]  # truncate generation
        return new_note_list
    return note_list



def pitch_range(midi_file):
    highest_pitch_note = None
    lowest_pitch_note = None
    for my_note in midi_file.flatten().notes:
        if my_note.isChord:
            return None
        if highest_pitch_note is None or lowest_pitch_note is None:
            highest_pitch_note = note.Note(my_note.nameWithOctave)
            lowest_pitch_note = note.Note(my_note.nameWithOctave) #note.Note(my_note.name)
            continue
        if my_note.pitch.frequency > highest_pitch_note.pitch.frequency:
            highest_pitch_note = note.Note(my_note.nameWithOctave)
        elif my_note.pitch.frequency < lowest_pitch_note.pitch.frequency:
            lowest_pitch_note = note.Note(my_note.nameWithOctave)
    pr = interval.Interval(lowest_pitch_note, highest_pitch_note).semitones
    return pr



def used_pitch_class(midi_file, chords_for_file = None):
    if chords_for_file is None:
        my_notes = grab_notes(midi_file)
    else:
        my_notes = grab_notes_trimmed(midi_file, chords_for_file)
    upc = set()
    for my_note in my_notes:
        if my_note== "c" or my_note== "r":
            continue
        upc.add(my_note)
    normalized_upc = len(upc) / len(my_notes)
    return normalized_upc

def unique_melody_bass_pitch_ratio(midi_file_bass, midi_file_melody, chords_for_file = None):
    if chords_for_file is None:
        bass_notes = grab_notes(midi_file_bass)
        melody_notes = grab_notes(midi_file_melody)
        trimmed_length = min(len(melody_notes), len(bass_notes))
        bass_notes = bass_notes[:trimmed_length]
        melody_notes = melody_notes[:trimmed_length]
    else:
        bass_notes = grab_notes_trimmed(midi_file_bass, chords_for_file)
        melody_notes = grab_notes_trimmed(midi_file_melody, chords_for_file)
    unique_count = 0
    for i in range(len(melody_notes)):
        if melody_notes[i] !=bass_notes[i]:
            unique_count += 1
    prop_unique = unique_count / len(melody_notes)
    return prop_unique

def unique_pitches_per_piece(midi_file):
    flat_notes = midi_file.flatten()#.notes
    pitch_set = set()
    for element in flat_notes:
        if isinstance(element, note.Note):
            pitch_set.add(str(element.fullName))
    number_pitches = len(pitch_set)
    return number_pitches

def pitch_interval(midi_file):
    # the average value of the interval between two consecutive pitches in semitones.
    notes = midi_file.flatten().notes
    intervals = []
    for i in range(0, len(notes)-1):
        j = i+1
        note_1 = notes[i]
        note_2 = notes[j]
        if note_1.isChord or note_2.isChord:
            #intervals.append(None)
            continue
        else:
            my_interval = abs(interval.Interval(note_1, note_2).semitones)
            intervals.append(my_interval)
    if len(intervals) == 0:
        return None
    avg = sum(intervals)/len(intervals)
    return avg

def pitch_class_entropy(midi_file):
    pitch_class_counts = analysis.pitchAnalysis.pitchAttributeCount(midi_file, pitchAttr='pitchClass')
    pitch_class_counts_updated = {}
    total = sum(pitch_class_counts.values())
    if total == 0:
        return None
    for i in range(12):
        pitch_class_name = pitch.Pitch(pitchClass=i).name  # Convert pitch class to note name
        count = pitch_class_counts.get(i, 0)  # Get count, default to 0 if not present
        pitch_class_counts_updated[pitch_class_name] = count/total

    entropy = 0
    for key, value in pitch_class_counts_updated.items():
        if value > 0:
            h = value * math.log(value)
            entropy += h
    entropy = -1 * entropy
    return entropy

def simple_chord_tone_ratio(midi_file, chords_for_file):
    chord_tone_count = 0
    non_chord_tone_count = 0
    notes_in_file = grab_notes_trimmed(midi_file, chords_for_file)
    for note, chord in zip(notes_in_file, chords_for_file):
        if note!= "r" and note != "c" and chord!= "NA":
            if note in chord_to_note_dict[chord]:
                chord_tone_count += 1
            else:
                non_chord_tone_count += 1
    if non_chord_tone_count + chord_tone_count == 0:
        return None
    return chord_tone_count / (non_chord_tone_count + chord_tone_count)

def complex_chord_tone_ratio(midi_file, chords_for_file):
    # the number of a subset of non-chord tones that are two semitones within the notes which are right after them
    chord_tone_count = 0
    non_chord_tone_count = 0
    non_chord_tone_proper = 0
    notes_in_file = grab_notes_trimmed(midi_file, chords_for_file, with_octave=True)
    for i in range(0, len(chords_for_file)):
        my_note = notes_in_file[i]
        if my_note != "r" and my_note != "c" and my_note != "NA" and chords_for_file[i]!= "NA":
            my_note_short = my_note[:-1]
            if my_note_short in chord_to_note_dict[chords_for_file[i]]:
                chord_tone_count += 1
            elif i < len(chords_for_file) - 1:
                next_note = notes_in_file[i+1]
                if next_note != "r" and next_note != "c" and next_note != "NA":
                    my_note_obj = note.Note(nameWithOctave=my_note)
                    next_note_obj = note.Note(nameWithOctave=next_note)
                    my_interval = abs(interval.Interval(my_note_obj, next_note_obj).semitones)
                    if my_interval <= 2:
                        non_chord_tone_proper+=1
        else:
            non_chord_tone_count+=1
    ctrp = (chord_tone_count + non_chord_tone_proper)/len(chords_for_file)
    return ctrp

def process_notes_2(generated_midis_locations):
    average_upcs = []
    average_ctrs = []
    average_prs = []
    for sample_location in generated_midis_locations:
        files = os.listdir(sample_location)
        # data = pd.read_csv(test_csv_location)
        sample_upcs = []
        sample_ctrs = []
        sample_prs = []
        for file in files:
            filename = sample_location + file
            # chords = row['chord'].split(" ")
            try:
                midi_file = converter.parse(filename)
            except Exception as e:
                print(filename)
                continue

            # pitch metrics
            upc = used_pitch_class(midi_file)
            sample_upcs.append(upc)
            pr = pitch_range(midi_file)
            sample_prs.append(pr)

        sample_upcs = [item for item in sample_upcs if item is not None]
        sample_prs = [item for item in sample_prs if item is not None]
        avg_upcs = sum(sample_upcs)/len(sample_upcs)
        avg_prs = sum(sample_prs) / len(sample_prs)
        average_upcs.append(avg_upcs)
        average_prs.append(avg_prs)
    #res = pd.DataFrame({'files': generated_midis_locations, 'upc': average_upcs, 'pr': average_prs, 'ctr': average_ctrs})
    #print(res)
    print("")


def process_notes(test_csv_location, midis_locations, file_suffix, nothing_prior = False):
    average_upcs = []
    average_ctrs = []
    average_prs = []
    average_unique = []
    average_pi = []
    average_entr = []
    average_ctrps = []
    average_length_diffs = []
    for sample_location in midis_locations:
        data = pd.read_csv(test_csv_location)
        sample_upcs = []
        sample_ctrs = []
        sample_prs = []
        sample_unique = []
        sample_pi = []
        sample_entr = []
        sample_ctrps = []
        sample_length_diffs = []
        for idx, row in data.iterrows():
            if nothing_prior:
                filename = f'{sample_location}track_{idx+1}.mid'
            else:
                filename = sample_location + row['long_name'] + file_suffix
            chords = row['chord'].split(" ")
            try:
                midi_file = converter.parse(filename)
            except Exception as e:
                print(filename)
                continue

            # length
            if not nothing_prior:
                the_notes = grab_notes(midi_file)
                diff = len(chords)-len(the_notes)
            else:
                diff = None
            sample_length_diffs.append(diff)


            # pitch metrics
            if not nothing_prior:
                upc = used_pitch_class(midi_file, chords)
            else:
                upc = used_pitch_class(midi_file)
            sample_upcs.append(upc)

            pr = pitch_range(midi_file)
            sample_prs.append(pr)

            unique = unique_pitches_per_piece(midi_file)
            sample_unique.append(unique)

            pi = pitch_interval(midi_file)
            sample_pi.append(pi)

            entr = pitch_class_entropy(midi_file)
            sample_entr.append(entr)

            # chord metrics
            if not nothing_prior:
                ctr = simple_chord_tone_ratio(midi_file, chords)
                ctrp = complex_chord_tone_ratio(midi_file, chords)
            else:
                ctr = None
                ctrp = None
            sample_ctrs.append(ctr)
            sample_ctrps.append(ctrp)

        # print("hi")
        data['length_diff'] = sample_length_diffs
        data['upc'] = sample_upcs
        data['pr'] = sample_prs
        data['unique_pitches'] = sample_unique
        data['pitch_interval'] = sample_pi
        data['pitch_entropy'] = sample_entr
        data['ctr'] = sample_ctrs
        data['ctrp'] = sample_ctrps
        if '/bass/' in sample_location:
            results_name = 'result_analyses/' + sample_location.split('/')[0] + "_bass_results.csv"
        elif '/melody/' in sample_location:
            results_name = 'result_analyses/' + sample_location.split('/')[0] + "_melody_results.csv"
        else:
            results_name = 'result_analyses/' + sample_location.split('/')[0] + "_results.csv"
        data.to_csv(results_name, index=False)
        sample_upcs = [item for item in sample_upcs if item is not None]
        sample_prs = [item for item in sample_prs if item is not None]
        sample_ctrs = [item for item in sample_ctrs if item is not None]
        sample_unique = [item for item in sample_unique if item is not None]
        sample_length_diffs = [item for item in sample_length_diffs if item is not None]
        sample_pi = [item for item in sample_pi if item is not None]
        sample_entr = [item for item in sample_entr if item is not None]
        avg_upcs = sum(sample_upcs)/len(sample_upcs)
        avg_prs = sum(sample_prs) / len(sample_prs)
        avg_unique = sum(sample_unique)/len(sample_unique)
        avg_pi = sum(sample_pi)/len(sample_pi)
        avg_entr = sum(sample_entr)/len(sample_entr)
        if not nothing_prior:
            avg_ctrs = sum(sample_ctrs) / len(sample_ctrs)
            avg_ctrps = sum(sample_ctrps)/len(sample_ctrps)
            avg_length_diffs = sum(sample_length_diffs) / len(sample_length_diffs)
        else:
            avg_ctrs = None
            avg_ctrps = None
            avg_length_diffs = None
        average_upcs.append(avg_upcs)
        average_ctrs.append(avg_ctrs)
        average_prs.append(avg_prs)
        average_unique.append(avg_unique)
        average_pi.append(avg_pi)
        average_entr.append(avg_entr)
        average_ctrps.append(avg_ctrps)
        average_length_diffs.append(avg_length_diffs)
    res = pd.DataFrame({'files': midis_locations,
                        'length_diff':average_length_diffs,
                        'upc': average_upcs,
                        'pr': average_prs,
                        'unique_pitches': average_unique,
                        'pitch_interval': average_pi,
                        'pitch_entropy': average_entr,
                        'ctr': average_ctrs,
                        'ctrp': average_ctrps},
                    )
    print(res)
    print("")
    return res

def add_unique_pitch_both(test_csv_location, midis_bass_location, midis_melody_location, file_suffix_bass, file_suffix_melody, model_name, nothing_prior = False):
    data = pd.read_csv(test_csv_location)
    unique_pitches_both = []
    for idx, row in data.iterrows():
        chords = row['chord'].split(" ")
        if nothing_prior:
            bass_midi_file = converter.parse(f'{midis_bass_location}track_{idx+1}{file_suffix_bass}')
            melody_midi_file = converter.parse(f'{midis_melody_location}track_{idx+1}{file_suffix_melody}')
        else:
            bass_midi_file = converter.parse(midis_bass_location + row['long_name'] + file_suffix_bass)
            melody_midi_file = converter.parse(midis_melody_location + row['long_name'] + file_suffix_melody)
        if not nothing_prior:
            unique_pitch_both = unique_melody_bass_pitch_ratio(bass_midi_file, melody_midi_file, chords)
        else:
            unique_pitch_both = unique_melody_bass_pitch_ratio(bass_midi_file, melody_midi_file)
        unique_pitches_both.append(unique_pitch_both)
    data[f'{model_name}_unique_pitch_ratio'] = unique_pitches_both
    average_unique_pitches_both = sum(unique_pitches_both)/len(unique_pitches_both)
    data = data[['long_name', f'{model_name}_unique_pitch_ratio']]
    return data, average_unique_pitches_both

# def process_notes(long_csv_location):
#     generated_results = pd.read_csv(long_csv_location)
#     notes_present_in_chord = 0
#     notes_not_present_in_chord = 0
#     ground_truth_notes_present_in_chord = 0
#     ground_truth_notes_not_present_in_chord = 0
#     for idx, row in generated_results.iterrows():
#         gen_note = row['generated_note']
#         ground_truth_note = row['ground_truth_note']
#         ground_truth_chord = row['chord']
#         note_in_chord = note_present_in_chord(note = gen_note, chord = ground_truth_chord)
#         ground_truth_note_in_chord = note_present_in_chord(note = ground_truth_note, chord = ground_truth_chord)
#         if note_in_chord:
#             notes_present_in_chord += 1
#         else:
#             notes_not_present_in_chord += 1
#         if ground_truth_note_in_chord:
#             ground_truth_notes_present_in_chord += 1
#         else:
#             ground_truth_notes_not_present_in_chord += 1
#     print(notes_present_in_chord)
#     print(notes_not_present_in_chord)
#     print(ground_truth_notes_present_in_chord)
#     print(ground_truth_notes_not_present_in_chord)
#     print(generated_results.head())


def main():
    generated_samples_2_4 = [
        'chord2melody_samples/generated_midis_400/',
        'chord2bass_samples/generated_midis_400/',
        'chord2sequentialbass_first_samples/generated_midis_400/melody/',
        'chord2sequentialbass_first_samples/generated_midis_400/bass/',
        'chord2sequentialmelody_first_samples/generated_midis_400/melody/',
        'chord2sequentialmelody_first_samples/generated_midis_400/bass/',
        'chord2joint_samples/generated_midis_400/melody/',
        'chord2joint_samples/generated_midis_400/bass/',
        'chord2jointdecoder_samples/generated_midis_400/melody/',
        'chord2jointdecoder_samples/generated_midis_400/bass/',
    ]
    generated_samples_1 = [
        "nothingprior2melody_samples/generated_midis_400/",
        "nothingprior2bass_samples/generated_midis_400/"
    ]
    nothing_prior_df = process_notes(test_csv_location="test_joint.csv",
                                     midis_locations=generated_samples_1,
                                     file_suffix="_generated.mid",
                                     nothing_prior=True)
    chord_generated_df = process_notes(test_csv_location ="test_joint.csv",
                                 midis_locations=generated_samples_2_4,
                                 file_suffix="_generated.mid")
    gt_bass_df = process_notes(test_csv_location="test_joint.csv",
                               midis_locations=['new_simplified_bass_files_c_midi/'],
                               file_suffix="_simplified_bass_c.mid")
    gt_melody_df = process_notes(test_csv_location = "test_joint.csv",
                                 midis_locations = ['new_simplified_melody_files_c_midi/'],
                                 file_suffix="_simplified_melody_c.mid")
    df = pd.concat([nothing_prior_df, gt_bass_df, gt_melody_df, chord_generated_df])
    df.to_csv("results.csv", index=False)
    print("")
    data_2, model_2 = add_unique_pitch_both(test_csv_location="test_joint.csv",
                          midis_bass_location='chord2bass_samples/generated_midis_400/',
                          midis_melody_location='chord2melody_samples/generated_midis_400/',
                          file_suffix_bass="_generated.mid",
                          file_suffix_melody="_generated.mid",
                                          model_name='model_2',)
    data_3a, model_3a = add_unique_pitch_both(test_csv_location="test_joint.csv",
                          midis_bass_location='chord2sequentialbass_first_samples/generated_midis_400/bass/',
                          midis_melody_location='chord2sequentialbass_first_samples/generated_midis_400/melody/',
                          file_suffix_bass="_generated.mid",
                          file_suffix_melody="_generated.mid",
                                          model_name='model_3a',)
    data_3b, model_3b = add_unique_pitch_both(test_csv_location="test_joint.csv",
                          midis_bass_location='chord2sequentialmelody_first_samples/generated_midis_400/bass/',
                          midis_melody_location='chord2sequentialmelody_first_samples/generated_midis_400/melody/',
                          file_suffix_bass="_generated.mid",
                          file_suffix_melody="_generated.mid",
                                          model_name='model_3b',)
    data_4a, model_4a = add_unique_pitch_both(test_csv_location="test_joint.csv",
                          midis_bass_location='chord2joint_samples/generated_midis_400/bass/',
                          midis_melody_location='chord2joint_samples/generated_midis_400/melody/',
                          file_suffix_bass="_generated.mid",
                          file_suffix_melody="_generated.mid",
                                          model_name='model_4a',)
    data_4b, model_4b = add_unique_pitch_both(test_csv_location="test_joint.csv",
                          midis_bass_location='chord2jointdecoder_samples/generated_midis_400/bass/',
                          midis_melody_location='chord2jointdecoder_samples/generated_midis_400/melody/',
                          file_suffix_bass="_generated.mid",
                          file_suffix_melody="_generated.mid",
                                          model_name='model_4b',)
    data_1, model_1 = add_unique_pitch_both(test_csv_location="test_joint.csv",
                          midis_bass_location='nothingprior2bass_samples/generated_midis_400/',
                          midis_melody_location='nothingprior2melody_samples/generated_midis_400/',
                          file_suffix_bass=".mid",
                          file_suffix_melody=".mid",
                          model_name='model_1',
                          nothing_prior=True)
    data_gt, gt = add_unique_pitch_both(test_csv_location="test_joint.csv",
                          midis_bass_location='new_simplified_bass_files_c_midi/',
                          midis_melody_location='new_simplified_melody_files_c_midi/',
                          file_suffix_bass="_simplified_bass_c.mid",
                          file_suffix_melody="_simplified_melody_c.mid",
                                          model_name='gt',)
    print("")
    df_unique_pitch_ratio = pd.concat([data_1, data_2, data_3a, data_3b, data_4a, data_4b, data_gt], axis=1)
    df_unique_pitch_ratio.to_csv("results_unique_pitch_ratio.csv", index=False)
    print("")

if __name__ == "__main__":
    os.makedirs('result_analyses', exist_ok=True)
    add_to_chord_to_note_dict()
    main()