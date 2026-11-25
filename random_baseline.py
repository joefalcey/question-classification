import data_reader
import evaluate

import random

class RandomChoice:
    def __init__(self):
        self.labels = []

    def train(self, train_data: list):
        self.labels = list(set(item.label for item in train_data))

    def predict_all(self, test_data: list):
        # Returns random labels for each test instance
        labels = self.labels
        return [random.choice(labels) for _ in test_data]

if __name__ == '__main__':
    train_data = data_reader.read('data/train_1000.label')
    test_data = data_reader.read('data/TREC_10.label')

    # Initialize and Train
    random_choice = RandomChoice()
    random_choice.train(train_data)

    predictions = random_choice.predict_all(test_data)

    print("\nFirst 10 predictions on the test set:")
    print('PREDICTIONS: ', predictions[:10])

    print("\nEvaluation of the random baseline:")
    true_labels = [item.label for item in test_data]
    evaluation = evaluate.evaluate(true_labels, predictions)
    print(evaluation)
