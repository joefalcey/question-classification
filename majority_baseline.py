import data_reader
import evaluate

class MajorityVote:
    def __init__(self):
        self.majority_label = None

    def train(self, training_data: list):
        label_count = dict()
        for item in training_data:
            label = item.label
            if label not in label_count:
                label_count[label] = 1
            else:
                label_count[label] += 1

        #print('LABEL COUNT: ', label_count)
        
        # Find the winner
        self.majority_label = max(label_count, key=label_count.get)
        # print('Majority class is: {self.majority_label}')

    def predict_all(self, test_data: list):
        # Blindly returns majority label for every test instance
        return [self.majority_label for _ in test_data]


if __name__ == '__main__':
    train_data = data_reader.read('data/train_1000.label')
    test_data = data_reader.read('data/TREC_10.label')

    # Initialize and Train
    majority_vote = MajorityVote()
    majority_vote.train(train_data)

    predictions = majority_vote.predict_all(test_data)

    print("\nFirst 10 predictions on the test set:")
    print(predictions[:10])

    print("\nEvaluation of the majority vote:")
    true_labels = [item.label for item in test_data]
    evaluation = evaluate.evaluate(true_labels, predictions)
    print(evaluation)
