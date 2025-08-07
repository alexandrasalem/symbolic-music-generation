# symbolic-music-generation

### Training
The two main training files are `train_model.py` and `train_model_no_chords.py`.

For `train_model.py` you need the MIDI files, the chord csv, and the chord tokenizer file.

For `train_model_no_chords.py` you just need to MIDI files.

In `train_model.py`, the model takes the chords as input, and generates the MIDI conditioned on the encoded chords. Causal masking is used on both the chord input and the target notes.

In `train_model_no_chords.py`, the model just generates MIDI.

#### To-Do
These two tasks will require making a new model class in `models.py`.
- Add a model that generates the bass and then afterwards the melody (instead of two separate models).
- Add a model that generates both the bass and melody at the same time--one note for each per chord.

### Generation
You generate new samples with `generate.py` and `generate_no_chords.py`.

You can evaluate the results with `evaluate.py`.