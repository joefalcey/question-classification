from collections import defaultdict
import argparse

import data_reader
from data_reader import Text

main_parser = data_reader.parser
parser = argparse.ArgumentParser(parents=[main_parser], add_help=False)
args = parser.parse_args()
if args.verbose: verbose = True

class Evaluation:
    def __init__(self, verbose: bool = False) -> None:
        self.metrics = []
        self.verbose = verbose
        self.accuracy = 0.0
        self.precision = 0.0
        self.recall = 0.0
        self.f1 = 0.0
        self.spec = 'macro-'

    def evaluate(self, true_labels: list[str], predicted_labels: list[str]) -> list:
        total_instances = len(true_labels)
        classes = set(true_labels)
        self.classes = classes
        tp_dict = defaultdict(lambda: defaultdict(int))
        error_dict = defaultdict(lambda: defaultdict(int))
        support_dict = defaultdict(int)
        correct_quant = 0

        for c in classes:
            tp_dict[c][c] = 0
            for sc in classes:
                if c != sc:
                    error_dict[c][sc] = 0

        for true_label, predicted_label in zip(true_labels, predicted_labels):
            if true_label == predicted_label:
                correct_quant += 1
                tp_dict[true_label][predicted_label] += 1
            else:
                error_dict[true_label][predicted_label] += 1

        precisions = []
        recalls = []
        f1s = []
        for c in classes:
            tp = tp_dict[c][c]
            fn = sum(error_dict[c].values())
            fp = sum(error_dict[other][c] for other in classes if other != c)
            support_dict[c] = tp + fn
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)
        return [classes, total_instances, correct_quant, precisions, recalls, f1s, support_dict]

    def calculations(self, eval_list: list) -> None:
        (classes, total_instances, correct_quant, precisions, recalls,
            f1s, support_dict) = eval_list
        num_classes = len(classes)
        self.accuracy = correct_quant / total_instances if total_instances > 0 else 0.0
        if self.spec == 'macro-':
            self.precision = sum(precisions) / num_classes if num_classes > 0 else 0.0
            self.recall = sum(recalls) / num_classes if num_classes > 0 else 0.0
            self.f1 = sum(f1s) / num_classes if num_classes > 0 else 0.0
        elif self.spec == 'micro-':
            tp_total = correct_quant
            fp_total = total_instances - correct_quant
            fn_total = total_instances - correct_quant
            micro_p = (tp_total / (tp_total + fp_total)
                if (tp_total + fp_total) > 0 else 0.0)
            micro_r = (tp_total / (tp_total + fn_total)
                if (tp_total + fn_total) > 0 else 0.0)
            micro_f1 = ((2 * micro_p * micro_r / (micro_p + micro_r))
                if (micro_p + micro_r) > 0 else 0.0)
            self.precision = micro_p
            self.recall = micro_r
            self.f1 = micro_f1
        elif self.spec == 'weighted ':
            weighted_p = sum(p * support_dict.get(c, 0)
                for c, p in zip(classes, precisions))
            weighted_r = sum(r * support_dict.get(c, 0)
                for c, r in zip(classes, recalls))
            weighted_f1 = sum(f * support_dict.get(c, 0)
                for c, f in zip(classes, f1s))
            self.precision = weighted_p / total_instances if total_instances > 0 else 0.0
            self.recall = weighted_r / total_instances if total_instances > 0 else 0.0
            self.f1 = weighted_f1 / total_instances if total_instances > 0 else 0.0
        else:
            raise ValueError(f"Unsupported metric spec: {self.spec!r}")

        self.metrics = [self.precision, self.recall, self.f1]

    def print_report(self, eval_list: list, model_name: str = "Model") -> None:
        print(model_name, ":")
        def print_metrics():
            spec = self.spec
            names = [f'{spec}precision',
                     f'{spec}recall', f'{spec}f1']
            for metric, name in zip(self.metrics, names):
                print(f'  {name} = {metric*100:.2f}%')
        self.spec = 'macro-'
        self.calculations(eval_list)
        print(f'  accuracy = {self.accuracy*100:.2f}%')
        print_metrics()
        if self.verbose:
            self.spec = 'micro-'
            self.calculations(eval_list)
            print_metrics()
            self.spec = 'weighted '
            self.calculations(eval_list)
            print_metrics()

if __name__ == '__main__':
    import pathlib
    from data_reader import read_data
    import os
    import majority_baseline
    import random_baseline
    import perceptron
    path = pathlib.Path(os.getcwd()) / 'data'
    train_file = read_data(path / 'train_5500.label')
    test_file = read_data(path / 'TREC_10.label')
    training_data = [Text(line) for line in train_file]
    test_data = [Text(line) for line in test_file]

    majority_vote = majority_baseline.MajorityVote()
    random_choice = random_baseline.RandomChoice()
    perceptron = perceptron.Perceptron()
    Models = [majority_vote, random_choice, perceptron,]

    for m in Models:
        m.train(training_data)
        predictions = m.predict(test_data)
        true_labels = [item.label[0] for item in test_data]
        evaluation = Evaluation(verbose=True)
        eval_list = evaluation.evaluate(true_labels, predictions)
        evaluation.print_report(eval_list, str(m.name))
