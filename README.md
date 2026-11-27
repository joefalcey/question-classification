# Question Classification Benchmark

This independent reimplementation of our Team Lab project is currently under active construction.

A benchmark repository comparing from-scratch multiclass perceptron classifiers against a fine-tuned BERT model and standard baselines for automated question classification.

Developed as part of the **Team Lab** module in the M.Sc. Computational Linguistics program at the University of Stuttgart (IMS).

## Overview

This project evaluates how lightweight linear models implemented from scratch compare to transformer-based architectures on multi-class question categorization, using standardized evaluation metrics and error analysis across question types.

### Models Included
* **Baselines:** Majority class and random classifiers
* **Linear Models (From Scratch):** Multiclass Perceptron and Averaged Perceptron
* **Transformer:** Fine-tuned BERT classifier (`PyTorch` / `Hugging Face`)

### Evaluation
The evaluation pipeline computes per-class and macro/weighted **Precision**, **Recall**, **F1-scores**, and **confusion matrices** to inspect misclassification patterns across question categories.

## Project Structure

> *Note: Repository layout is currently being unified across modules; integration and compatibility updates are in progress.*

```text
.
├── baselines/          # Majority and random baseline classifiers
├── perceptron/         # From-scratch multiclass & averaged perceptron implementations
├── bert/               # Fine-tuned BERT training and inference scripts
├── evaluation/         # Metrics (P/R/F1) and confusion matrix generation
└── data/               # Dataset preprocessing and splits
