import data_reader
from data_reader import Text
import argparse
from collections import Counter

main_parser = data_reader.parser
parser = argparse.ArgumentParser(parents=[main_parser], add_help=False)
args = parser.parse_args()
if args.verbose: verbose = True

class MajorityVote:
    def __init__(self, verbose=False):
        self.training_data = Text
        self.majority_label = str()
        self.verbose = verbose

    def train(self, training_data: list[Text]) -> None:
        if self.verbose: print(f"Training with {len(training_data)} samples...")
        label_list = [line.label[0] for line in training_data]
        label_counts = Counter(label_list)
        self.majority_label = Counter(label_list).most_common(1)[0][0]
        if args.verbose: print(f'\nCoarse label counts: ', label_counts, '\n')

    def predict(self, test_data: list[Text]) -> list[str]:
        # Blindly returns majority label for every test instance
        if args.verbose: print('Majority label: ', self.majority_label, '\n')
        return [self.majority_label for _ in test_data]

    def __repr__(self):
        return self.majority_label

    @property
    def name(self) -> str:
        return "MajorityVote"

if __name__ == '__main__':
    import pathlib
    import os
    from data_reader import read_data
    path = pathlib.Path(os.getcwd()) / 'data'
    train_file = read_data(path / 'train_5500.label')
    test_file = read_data(path / 'TREC_10.label')
    training_data = [Text(line) for line in train_file]
    test_data = [Text(line) for line in test_file]
    majority_vote = MajorityVote(verbose=True)
    majority_vote.train(training_data)
    predictions = majority_vote.predict(test_data)
    true_labels = [line.label[0] for line in training_data]
    print('Loading sample set...')
    for true, predicted in zip(true_labels[:10], predictions[:10]):
        print(f'true label={true}, predicted label={predicted}')
