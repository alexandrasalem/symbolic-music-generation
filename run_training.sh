#!/bin/bash

echo "running nothingprior2bass and nothingprior2melody AKA no chord (bass and melody, separately)"
python train_model_no_chords.py --bass_or_melody=bass --piece_or_theme=piece
python train_model_no_chords.py --bass_or_melody=bass --piece_or_theme=theme
python train_model_no_chords.py --bass_or_melody=melody --piece_or_theme=piece
python train_model_no_chords.py --bass_or_melody=melody --piece_or_theme=theme

echo "running chord2bass and chord2melody AKA chord independent (bass and melody, separately)"
python train_model.py --bass_or_melody=bass --piece_or_theme=piece
python train_model.py --bass_or_melody=bass --piece_or_theme=theme
python train_model.py --bass_or_melody=melody --piece_or_theme=piece
python train_model.py --bass_or_melody=melody --piece_or_theme=theme

echo "running chord2sequentialbass_first AKA chord bass first"
python train_sequential_model.py --bass_or_melody=bass --piece_or_theme=piece
python train_sequential_model.py --bass_or_melody=bass --piece_or_theme=theme

echo "running chord2sequentialmelody_first AKA chord melody first"
python train_sequential_model.py --bass_or_melody=melody --piece_or_theme=piece
python train_sequential_model.py --bass_or_melody=melody --piece_or_theme=theme

echo "running chord2jointdecoder AKA chord co-generate"
python train_joint_decoder_model.py --piece_or_theme=piece
python train_joint_decoder_model.py --piece_or_theme=theme

#echo "running chord2symmetric AKA NEW chord co-generate"
#python train_symmetric_decoder_model.py --piece_or_theme=piece
#python train_symmetric_decoder_model.py --piece_or_theme=theme

