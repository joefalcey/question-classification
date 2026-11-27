import argparse
import pathlib
import os

from data_reader import read_data, Text
import data_reader
# from evaluate import evaluate, print_detailed_report

# # BERT integration (optional - requires torch/transformers)
# try:
#     from bert_pipeline import BERTClassifier
#     BERT_AVAILABLE = True
# except ImportError:
#     BERT_AVAILABLE = False
#     print("Note: BERT not available. Install dependencies with: uv sync")

data_parser = data_reader.parser
parser = argparse.ArgumentParser(parents=[data_parser], add_help=False)
args = parser.parse_args()

path = pathlib.Path(os.getcwd()) / 'data'

def test_models():
    from random_baseline import RandomChoice
    from majority_baseline import MajorityVote
    from perceptron import Perceptron
    # from averaged_perceptron import AveragedPerceptron
    # from sklearn_models import SklearnClassifier
    # from hierarchical_perceptron import HierarchicalPerceptron
    # from hierarchical_svm import HierarchicalSVM
    # from hybrid_kmeans import KMeansHierarchical
    # Add BERT models if available
    # if BERT_AVAILABLE:
    #     bert_models = [
    #         ("BERT (DistilBERT, Balanced)", BERTClassifier, {
    #             "model_name": "distilbert-base-uncased",
    #             "class_weights": "balanced",
    #             "use_focal_loss": True,
    #             "batch_size": 16,
    #             "learning_rate": 2e-5,
    #         }),
    #     ]
    #     models_to_test.extend(bert_models)
    #     print("Note: BERT models added to evaluation. This will significantly increase runtime.")
    models_to_test = [
        # PERCEPTRON MODELS
        Perceptron(),
    ]

if __name__ == "__main__":
    train_file = read_data(path / 'train_5500.label')
    test_file = read_data(path / 'TREC_10.label')
    training_data = [Text(line) for line in train_file]
    test_data = [Text(line) for line in test_file]
    # print('\nTraining data:')
    for line in training_data:
        print(line)
