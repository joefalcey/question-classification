import argparse
import pathlib
from collections.abc import Iterator

import nltk
from nltk import WordNetLemmatizer as wnl

try:
    nltk.download('punkt_tab', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
except Exception as e:
    print(f'Error downloading NLTK package: {e}')

parser = argparse.ArgumentParser(
    description='Reads question data for classification prediction')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='Turns on verbose mode')
parser.add_argument('-l', '--lemmatization', action='store_true',
                    help='Lemmatizes tokenized data')
args = parser.parse_args()

def read_data(file: pathlib.Path) -> list:
    file_data = []
    with open(file, 'r', encoding='utf-8', errors="ignore") as f:
        for line in f.readlines():
            split_line = line.split(':', 1)
            coarse_label = split_line[0]
            fine_label = split_line[1].split(' ', 1)[0]
            text = split_line[1].split(' ', 1)[1]
            label = (coarse_label, fine_label)
            if not args.lemmatization:
                tokens = [token.lower() for token in text.split() if token.isalnum()
                          or '\'' in token]
            else:
                tokens = [wnl().lemmatize(token).lower() for token in text.split()
                          if token.isalnum() or '\'' in token]
            file_data.append([label, tokens])
    return file_data

class Text:
    def __init__(self, line: list):
        self.label, self.tokens = line

    def __iter__(self) -> Iterator[list]:
        yield self.label
        yield self.tokens

    def __len__(self) -> int:
        return len(self.tokens)

    def __repr__(self) -> str:
        return f"[{self.label!r}] {self.tokens!r}"

if __name__ == "__main__":
    import pathlib
    import os
    path = pathlib.Path(os.getcwd()) / 'data'
    train_file = path / 'train_5500.label'
    data = read_data(train_file)
    text = [Text(line) for line in data]
    print(text)
