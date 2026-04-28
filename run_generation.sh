#!/bin/bash

echo "making folders"
mkdir checkpoints
mkdir checkpoints/chord2bass_train_checkpoints
mkdir checkpoints/chord2melody_train_checkpoints
mkdir checkpoints/chord2bass_theme_train_checkpoints
mkdir checkpoints/chord2melody_theme_train_checkpoints

mkdir checkpoints/nothingprior2bass_train_checkpoints
mkdir checkpoints/nothingprior2melody_train_checkpoints
mkdir checkpoints/nothingprior2bass_theme_train_checkpoints
mkdir checkpoints/nothingprior2melody_theme_train_checkpoints

mkdir checkpoints/chord2sequentialbass_first_train_checkpoints
mkdir checkpoints/chord2sequentialmelody_first_train_checkpoints
mkdir checkpoints/chord2sequentialbass_first_theme_train_checkpoints
mkdir checkpoints/chord2sequentialmelody_first_theme_train_checkpoints

mkdir checkpoints/chord2symmetricdecoder_train_checkpoints
mkdir checkpoints/chord2symmetricdecoder_theme_train_checkpoints

mkdir checkpoints/chord2jointdecoder_train_checkpoints
mkdir checkpoints/chord2jointdecoder_theme_train_checkpoints

mkdir logs
mkdir samples


echo "pulling down log files"
rsync -avP alexandra@146.245.250.218:/scratch/alexandra/symbolic-music-generation/chord2bass_train_log.log ./logs
rsync -avP alexandra@146.245.250.218:/scratch/alexandra/symbolic-music-generation/chord2melody_train_log.log ./logs
rsync -avP alexandra@146.245.250.218:/scratch/alexandra/symbolic-music-generation/chord2bass_theme_train_log.log ./logs
rsync -avP alexandra@146.245.250.218:/scratch/alexandra/symbolic-music-generation/chord2melody_theme_train_log.log ./logs
rsync -avP alexandra@146.245.250.218:/scratch/alexandra/symbolic-music-generation/nothingprior2bass_train_log.log ./logs
rsync -avP alexandra@146.245.250.218:/scratch/alexandra/symbolic-music-generation/nothingprior2melody_train_log.log ./logs
rsync -avP alexandra@146.245.250.218:/scratch/alexandra/symbolic-music-generation/nothingprior2bass_theme_train_log.log ./logs
rsync -avP alexandra@146.245.250.218:/scratch/alexandra/symbolic-music-generation/nothingprior2melody_theme_train_log.log ./logs

rsync -avP alexandra@146.245.250.218:/scratch/alexandra/symbolic-music-generation/chord2sequentialbass_first_train_log.log ./logs
rsync -avP alexandra@146.245.250.218:/scratch/alexandra/symbolic-music-generation/chord2sequentialmelody_first_train_log.log ./logs
rsync -avP alexandra@146.245.250.218:/scratch/alexandra/symbolic-music-generation/chord2sequentialbass_first_theme_train_log.log ./logs
rsync -avP alexandra@146.245.250.218:/scratch/alexandra/symbolic-music-generation/chord2sequentialmelody_first_theme_train_log.log ./logs

rsync -avP alexandra@146.245.250.218:/scratch/alexandra/symbolic-music-generation/chord2symmetricdecoder_train_log.log ./logs
rsync -avP alexandra@146.245.250.218:/scratch/alexandra/symbolic-music-generation/chord2symmetricdecoder_theme_train_log.log ./logs

rsync -avP alexandra@146.245.250.218:/scratch/alexandra/symbolic-music-generation/chord2jointdecoder_train_log.log ./logs
rsync -avP alexandra@146.245.250.218:/scratch/alexandra/symbolic-music-generation/chord2jointdecoder_theme_train_log.log ./logs

echo "pulling down checkpoints"
rsync -avP alexandra@146.245.250.218:/scratch/alexandra/symbolic-music-generation/chord2bass_train_checkpoints/chord2bass_epoch_100.pt ./checkpoints/chord2bass_train_checkpoints
rsync -avP alexandra@146.245.250.218:/scratch/alexandra/symbolic-music-generation/chord2melody_train_checkpoints/chord2melody_epoch_100.pt ./checkpoints/chord2melody_train_checkpoints
rsync -avP alexandra@146.245.250.218:/scratch/alexandra/symbolic-music-generation/chord2bass_theme_train_checkpoints/chord2bass_theme_epoch_100.pt ./checkpoints/chord2bass_theme_train_checkpoints
rsync -avP alexandra@146.245.250.218:/scratch/alexandra/symbolic-music-generation/chord2melody_theme_train_checkpoints/chord2melody_theme_epoch_100.pt ./checkpoints/chord2melody_theme_train_checkpoints

rsync -avP alexandra@146.245.250.218:/scratch/alexandra/symbolic-music-generation/nothingprior2bass_train_checkpoints/nothingprior2bass_epoch_100.pt ./checkpoints/nothingprior2bass_train_checkpoints
rsync -avP alexandra@146.245.250.218:/scratch/alexandra/symbolic-music-generation/nothingprior2melody_train_checkpoints/nothingprior2melody_epoch_100.pt ./checkpoints/nothingprior2melody_train_checkpoints
rsync -avP alexandra@146.245.250.218:/scratch/alexandra/symbolic-music-generation/nothingprior2bass_theme_train_checkpoints/nothingprior2bass_theme_epoch_100.pt ./checkpoints/nothingprior2bass_theme_train_checkpoints
rsync -avP alexandra@146.245.250.218:/scratch/alexandra/symbolic-music-generation/nothingprior2melody_theme_train_checkpoints/nothingprior2melody_theme_epoch_100.pt ./checkpoints/nothingprior2melody_theme_train_checkpoints

rsync -avP alexandra@146.245.250.218:/scratch/alexandra/symbolic-music-generation/chord2sequentialbass_first_train_checkpoints/chord2sequentialbass_first_epoch_100.pt ./checkpoints/chord2sequentialbass_first_train_checkpoints
rsync -avP alexandra@146.245.250.218:/scratch/alexandra/symbolic-music-generation/chord2sequentialmelody_first_train_checkpoints/chord2sequentialmelody_first_epoch_100.pt ./checkpoints/chord2sequentialmelody_first_train_checkpoints
rsync -avP alexandra@146.245.250.218:/scratch/alexandra/symbolic-music-generation/chord2sequentialbass_first_theme_train_checkpoints/chord2sequentialbass_first_theme_epoch_100.pt ./checkpoints/chord2sequentialbass_first_theme_train_checkpoints
rsync -avP alexandra@146.245.250.218:/scratch/alexandra/symbolic-music-generation/chord2sequentialmelody_first_theme_train_checkpoints/chord2sequentialmelody_first_theme_epoch_100.pt ./checkpoints/chord2sequentialmelody_first_theme_train_checkpoints

rsync -avP alexandra@146.245.250.218:/scratch/alexandra/symbolic-music-generation/chord2symmetricdecoder_train_checkpoints/chord2symmetricdecoder_epoch_100.pt ./checkpoints/chord2symmetricdecoder_train_checkpoints
rsync -avP alexandra@146.245.250.218:/scratch/alexandra/symbolic-music-generation/chord2symmetricdecoder_theme_train_checkpoints/chord2symmetricdecoder_theme_epoch_100.pt ./checkpoints/chord2symmetricdecoder_theme_train_checkpoints

rsync -avP alexandra@146.245.250.218:/scratch/alexandra/symbolic-music-generation/chord2jointdecoder_train_checkpoints/chord2jointdecoder_epoch_100.pt ./checkpoints/chord2jointdecoder_train_checkpoints
rsync -avP alexandra@146.245.250.218:/scratch/alexandra/symbolic-music-generation/chord2jointdecoder_theme_train_checkpoints/chord2jointdecoder_theme_epoch_100.pt ./checkpoints/chord2jointdecoder_theme_train_checkpoints

echo "generating samples from checkpoints"
python generate_no_chords.py --piece_or_theme=piece --bass_or_melody=bass --epoch=100
python generate_no_chords.py --piece_or_theme=theme --bass_or_melody=bass --epoch=100
python generate_no_chords.py --piece_or_theme=piece --bass_or_melody=melody --epoch=100
python generate_no_chords.py --piece_or_theme=theme --bass_or_melody=melody --epoch=100

python generate.py --piece_or_theme=piece --bass_or_melody=bass --epoch=100
python generate.py --piece_or_theme=theme --bass_or_melody=bass --epoch=100
python generate.py --piece_or_theme=piece --bass_or_melody=melody --epoch=100
python generate.py --piece_or_theme=theme --bass_or_melody=melody --epoch=100

python generate_sequential.py --piece_or_theme=piece --bass_or_melody=bass --epoch=100
python generate_sequential.py --piece_or_theme=theme --bass_or_melody=bass --epoch=100
python generate_sequential.py --piece_or_theme=piece --bass_or_melody=melody --epoch=100
python generate_sequential.py --piece_or_theme=theme --bass_or_melody=melody --epoch=100

python generate_symmetric.py --piece_or_theme=piece --epoch=100
python generate_symmetric.py --piece_or_theme=theme --epoch=100

python generate_jointdecoder.py --piece_or_theme=piece --epoch=100
python generate_jointdecoder.py --piece_or_theme=theme --epoch=100

echo "combining samples"

python combine_midi.py --bass_location=samples/chord2symmetricdecoder_theme_samples/generated_midis_100/bass
--melody_location=samples/chord2symmetricdecoder_theme_samples/generated_midis_100/melody
--combined_location=samples/chord2symmetricdecoder_theme_samples/generated_midis_100/combined
--test_csv_location=test_joint_themes_held_out.csv

python combine_midi.py --bass_location=samples/chord2sequentialmelody_first_samples/generated_midis_100/bass
--melody_location=samples/chord2sequentialmelody_first_samples/generated_midis_100/melody
--combined_location=samples/chord2sequentialmelody_first_samples/generated_midis_100/combined
--test_csv_location=test_joint.csv