# symbolic-music-generation

### Training
There are four training files and four corresponding generate files, for the five separate models.
- `train_model_no_chords.py`: No Chord
- `train_model.py`: Chord Independent (Bass and Melody models separate)
- `train_sequential_model.py`: Chord Bass/Melody 1st
- `train_joint_decoder_model.py`: Chord Co-Generate (one decoder, two linear heads)


### Generation
Generation is done with `generate.py` and the other similar files.

Evaluation is performed with `metrics.py` and `t-tests.py`.

### Samples

A small selection of generated files are in `samples/`.