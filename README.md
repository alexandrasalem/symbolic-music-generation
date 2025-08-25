# symbolic-music-generation

### Training
There are five training files and five corresponding generate files, for the five separate models.
- `train_model_no_chords.py`: No Chord
- `train_model.py`: Chord Independent (Bass and Melody models separate)
- `train_sequential_model.py`: Chord Bass/Melody First
- `train_joint_model.py`: Chord Co-Generate 1 (separate decoders, loss of both added)
- `train_joint_decoder_model.py`: Chord Co-Generate 2 (one decoder, two linear heads)


### Generation
You generate new samples with `generate.py` and the other similar files.

You can evaluate the results with `metrics.py` and `t-tests.py`.