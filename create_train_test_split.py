import pandas as pd

chord_csv_name = "chords_edited.csv"

def create_train_test_split(chord_csv):
    full_df = pd.read_csv(chord_csv)
    pieces = []
    for _, row in full_df.iterrows():
        piece = row["long_name"].split("_")[0]
        pieces.append(piece)
    full_df['piece'] = pieces

    test_pieces = ["M455", "B075"]

    train_df = full_df[-full_df["piece"].isin(test_pieces)]
    test_df = full_df[full_df["piece"].isin(test_pieces)]

    train_name = f"train_{chord_csv_name}"
    test_name = f"test_{chord_csv_name}"

    train_df.to_csv(train_name, index=False)
    test_df.to_csv(test_name, index=False)

if __name__ == "__main__":
    create_train_test_split(chord_csv_name)