"""Download the UCI Spambase dataset and save it as data/spambase.csv (with header + 'spam' target)."""
import os
import sys
import pandas as pd

WORDS = ["make", "address", "all", "3d", "our", "over", "remove", "internet", "order", "mail",
         "receive", "will", "people", "report", "addresses", "free", "business", "email", "you",
         "credit", "your", "font", "000", "money", "hp", "hpl", "george", "650", "lab", "labs",
         "telnet", "857", "data", "415", "85", "technology", "1999", "parts", "pm", "direct", "cs",
         "meeting", "original", "project", "re", "edu", "table", "conference"]
CHARS = [";", "(", "[", "!", "$", "#"]
COLUMNS = (["word_freq_" + w for w in WORDS]
           + ["char_freq_" + c for c in CHARS]
           + ["capital_run_length_average", "capital_run_length_longest", "capital_run_length_total", "spam"])

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
OUT = os.path.join("data", "spambase.csv")


def main():
    os.makedirs("data", exist_ok=True)
    try:
        df = pd.read_csv(URL, header=None)
    except Exception as err:
        print("UCI download failed:", err)
        print("Manual fallback: download spambase.data from https://archive.ics.uci.edu/dataset/94/spambase,")
        print("save it next to this script, then run:  python download_data.py spambase.data")
        if len(sys.argv) > 1:
            df = pd.read_csv(sys.argv[1], header=None)
        else:
            sys.exit(1)
    assert df.shape[1] == 58, f"expected 58 columns, got {df.shape[1]}"
    df.columns = COLUMNS
    df.to_csv(OUT, index=False)
    print("Saved", OUT, df.shape)
    print(df["spam"].value_counts())


if __name__ == "__main__":
    main()
