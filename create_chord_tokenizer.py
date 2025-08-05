from tokenizers import Tokenizer, processors, decoders
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import WhitespaceSplit
import pandas as pd

# Initialize an empty tokenizer
tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
tokenizer.pre_tokenizer = WhitespaceSplit()

print(tokenizer.pre_tokenizer.pre_tokenize_str("C:min G:maj/3 G:7/3 C:min"))
# Trainer with special tokens
trainer = WordLevelTrainer(
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    min_frequency=1,
)

# Train from file
tokenizer.train(["test_chords_edited.txt"], trainer)

encoding = tokenizer.encode("C:min G:maj G:maj G:maj C:min")
print(encoding.tokens)

cls_token_id = tokenizer.token_to_id("[CLS]")
sep_token_id = tokenizer.token_to_id("[SEP]")
print(cls_token_id, sep_token_id)

tokenizer.post_processor = processors.TemplateProcessing(
    single=f"[CLS]:0 $A:0 [SEP]:0",
    pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
    special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)],
)

tokenizer.decoder = decoders.WordPiece(prefix='##')


# saving vocab
sorted_items = dict(sorted(tokenizer.get_vocab().items()))  # returns a list of tuples
df = pd.DataFrame.from_dict(sorted_items, orient='index', columns=['count'])
df['chord'] = df.index
df = df.loc[:, ['chord']]
df.to_csv("chords.csv", index=False)

tokenizer.save("test_chord_tokenizer.json")

chord_tokenizer = Tokenizer.from_file("test_chord_tokenizer.json")
print(chord_tokenizer.encode('C:min NA G:maj G:maj C:min').tokens)