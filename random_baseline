import random
import argparse
import data_reader
from data_reader import Text

main_parser = data_reader.parser
parser = argparse.ArgumentParser(parents=[main_parser], add_help=False)
args = parser.parse_args()
if args.verbose: verbose = True

class RandomChoice:
    def __init__(self, verbose = False):
        self.label_list = []
        self.verbose = verbose

    def train(self, training_data: list[Text]) -> None:
        if self.verbose: print(f"Training with {len(training_data)} samples...")
        self.label_list = [line.label[0] for line in training_data]

    def predict(self, test_data: list[Text]) -> list:
        # Returns random labels for each test instance
        return [random.choice(self.label_list) for _ in test_data]

    @property
    def name(self) -> str:
        return "RandomChoice"

if __name__ == '__main__':
    import pathlib
    import os
    from data_reader import read_data
    path = pathlib.Path(os.getcwd()) / 'data'
    train_file = read_data(path / 'train_5500.label')
    test_file = read_data(path / 'TREC_10.label')
    training_data = [Text(line) for line in train_file]
    test_data = [Text(line) for line in test_file]
    random_choice = RandomChoice(verbose=True)
    random_choice.train(training_data)
    predictions = random_choice.predict(test_data)
    print(predictions[:10])
