import pandas as pd
import re

chord_csv_name = "chords_edited-key-tranposed.csv"

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

def create_train_test_split_themes(chord_csv):
    full_df = pd.read_csv(chord_csv)
    themes_or_vars = []
    for _, row in full_df.iterrows():
        theme_or_var = row["long_name"].split("_")[1]
        themes_or_vars.append(theme_or_var)
    full_df['theme_or_var'] = themes_or_vars

    test_pieces = ["00"]

    train_df = full_df[-full_df["theme_or_var"].isin(test_pieces)]
    test_df = full_df[full_df["theme_or_var"].isin(test_pieces)]

    train_name = f"train_themes_held_out_{chord_csv_name}"
    test_name = f"test_themes_held_out_{chord_csv_name}"

    train_df.to_csv(train_name, index=False)
    test_df.to_csv(test_name, index=False)

if __name__ == "__main__":
    create_train_test_split(chord_csv_name)
    create_train_test_split_themes(chord_csv_name)