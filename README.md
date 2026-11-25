# Question Type Classification

Developed as part of the Team Lab module in the M.Sc. Computational Linguistics program at the University of Stuttgart (IMS) alongside Iftekhar Ahmed Uday.

This repository contains an NLP pipeline designed to classify open-domain English questions into semantic categories using the **TREC Question Classification dataset**.

The project explores a progression of machine learning architectures—ranging from baseline models and custom Perceptrons to Scikit-Learn SVMs and BERT with data augmentation—finding that **proper data augmentation enables neural networks to significantly outperform linear approaches**.

**Current Status**: BERT with comprehensive data augmentation achieves **87.40% test accuracy** (raw text), surpassing all previous methods including Flat SVM (67.41%). Lemmatized text achieves 85.80%.

> **🎯 Key Finding**: Addressing the severe class imbalance (794:1 → 7.6:1) through data augmentation enabled **all models to improve significantly**, with neural networks showing the largest gains (+40-55% for BERT).

## 📊 Dataset Overview

The model is trained on the TREC dataset, which features a two-tiered taxonomy:
* **6 Coarse Classes:** `ABBR` (Abbreviation), `DESC` (Description), `ENTY` (Entity), `HUM` (Human), `LOC` (Location), `NUM` (Numeric).
* **50 Fine-grained Classes:** (e.g., `DESC:def`, `LOC:city`, `NUM:count`).

* **Original Class Imbalance:** 794:1 ratio between most frequent (`HUM:ind` - 794 examples) and rarest (`ENTY:currency` - 1 example) classes.

## 🚀 Project Architecture

The codebase is highly modular, allowing for seamless toggling of feature extraction methods and model types.

### 1. Feature Engineering
* **Bag-of-Words (BoW):** Unigram and Bigram extraction.
* **TF-IDF:** Term Frequency-Inverse Document Frequency weighting.
* **Dense Embeddings:** 300-D GloVe embeddings with mean-pooling.
* **Text Processing:** Optional NLTK-based Lemmatization.

### 2. Core Models
* **Baselines:** Majority Class Vote and Random Choice.
* **Perceptrons:** Custom-built Standard and Averaged Perceptrons.
* **Scikit-Learn Models:** Linear Support Vector Classification (`LinearSVC`) and Random Forest.
* **BERT Pipeline:** DistilBERT with data augmentation ([bert_pipeline/](bert_pipeline/))

### 3. Hierarchical Orchestration
To tackle the fine-grained classification challenge, we implemented a "Manager-Expert" pattern:
* **Hard-Coded Hierarchy:** A Master model predicts the coarse label, and routes the text to 1 of 6 Expert models trained specifically on that category's fine-grained labels.
* **Hybrid K-Means Sub-Clustering:** A 3-tier system that predicts the coarse class, applies K-Means *within* that class to handle label bloat, and routes to hyper-specialized sub-experts.

## 📈 Key Findings & Evaluation

### Current Best Result

**BERT with comprehensive data augmentation** significantly outperforms all previous methods:

| Model | Test Accuracy | Macro F1 | Configuration |
|-------|---------------|----------|---------------|
| **BERT + Augmentation (Raw)** | **87.40%** | **79.45%** | DistilBERT, 10.2K samples, no lemmatization |
| **BERT + Augmentation (Lemmatized)** | **85.80%** | **76.40%** | DistilBERT, 10.2K samples, lemmatized |

---

### 📊 Complete Results Comparison

#### Original Dataset (5,452 samples, 794:1 imbalance)

| Model | Test Accuracy | Macro F1 | Configuration |
|-------|---------------|----------|---------------|
| **Linear Models** |||
| Flat SVM (BoW Unigram) | 67.41% | ~50-55% | Lemmatized |
| Avg. Perceptron (BoW Unigram) | 64.41% | ~45-50% | Lemmatized |
| Hierarchical SVM | 67.44% | ~50-55% | Coarse → Expert |
| Hierarchical Perceptron | 58.95% | ~45-50% | Cascading errors |
| K-Means SVM (k=4) | 64.80% | ~45-50% | Clustering approach |
| **Other Approaches** |||
| MLP (BoW Unigram) | 57.92% | ~40-45% | Overfitting |
| Dense Embeddings (GloVe) | ~57% | ~40-45% | Washes out triggers |

#### Augmented Dataset (10,233 samples, 7.6:1 imbalance)

| Model | Test Accuracy | Macro F1 | Configuration |
|-------|---------------|----------|---------------|
| **BERT Approaches** |||
| BERT + Augmentation (Raw) | **87.40%** | **79.45%** | DistilBERT, no lemmatization ✅ |
| BERT + Augmentation (Lemmatized) | **85.80%** | **76.40%** | DistilBERT, lemmatized |
| **Linear Models** |||
| Hierarchical SVM (BoW) | **74.52%** | **74.72%** | Best linear on augmented ✅ |
| Flat SVM (BoW Unigram) | **74.52%** | ~70-75% | Lemmatized |
| Avg. Perceptron (BoW) | **69.80%** | ~65-70% | Lemmatized |
| MLP (BoW Unigram) | **68.14%** | ~60-65% | Great improvement |
| K-Means SVM (k=4) | **71.35%** | ~68-72% | Clustering approach |

---

### Key Improvements from Augmentation

| Model | Original (5.4K) | Augmented (10.2K) | Improvement |
|-------|-----------------|-------------------|-------------|
| Flat SVM | 67.41% | **74.52%** | +7.1% |
| Hierarchical SVM | 67.44% | **74.72%** | +7.3% |
| Avg. Perceptron | 64.41% | **69.80%** | +5.4% |
| MLP | 57.92% | **68.14%** | +10.2% |
| BERT (Raw) | 32-48% | **87.40%** | +40-55% |

**All models benefited from augmentation**, with neural networks (MLP, BERT) showing the largest gains.

### Key Insights

1. **Data Augmentation Benefits All Models**: Reducing imbalance from 794:1 to 7.6:1 improved every model:
   - Linear models: +5-10% (SVM: 67% → 75%, MLP: 58% → 68%)
   - Neural networks: +40-55% (BERT: 32-48% → 87%)
   - Hierarchical approaches became viable (previously failed due to cascading errors)

2. **Raw Text vs Lemmatization**:
   - **BERT**: Raw text performs better (87.40% vs 85.80%) — BERT's WordPiece and contextual embeddings handle morphology internally
   - **Linear models**: Lemmatized text performs better — normalization helps combine sparse features

3. **Sample Quality and Quantity**: Both matter. Comprehensive augmentation (10,233 samples) significantly outperformed partial augmentation (5,787 samples), demonstrating that class balance AND total sample count are important.

4. **Why BERT Dominates With Balance**: With severe imbalance (794:1), trigger word matching (linear models) wins at 67%. With reasonable balance (7.6:1), semantic understanding (BERT) dominates at 87%.

5. **Hierarchical Approaches**: Failed on original data (cascading errors) but became competitive on augmented data (74.72% F1).

## 🤖 BERT Training with Data Augmentation

### Augmentation Pipeline

The BERT pipeline ([bert_pipeline/](bert_pipeline/)) includes comprehensive data augmentation:

**Augmentation Process:**
1. **Synthetic Generation**: Generated ~1,345 synthetic questions for rare classes (< 20 samples) using LLM prompts
2. **Back-Translation**: Applied paraphrase augmentation via German/French/Spanish translation for middle-tail classes (20-100 samples)
3. **Strategic Downsampling**: Capped majority classes at 552 samples to prevent over-representation

**Augmented Dataset Statistics:**

| Metric | Original | Augmented |
|--------|----------|-----------|
| Total Samples | 5,452 | **10,233** |
| Imbalance Ratio | 240.5:1 | **7.6:1** ✅ |
| Range | 4 - 962 samples | **73 - 552 samples** ✅ |
| Rarest Class | 4 samples | **73 samples** ✅ |

### Quick Start

```bash
# Train BERT on augmented dataset (raw text - best performance)
uv run python -m bert_pipeline.run_bert --no-lemmatize

# Train with lemmatized text (slightly lower accuracy)
uv run python -m bert_pipeline.run_bert

# Evaluate all linear models on augmented data (5-run comparison)
uv run main.py

# Compare multiple BERT configurations
uv run python -m bert_pipeline.run_bert --compare

# Run multiple experiments with different seeds
uv run python -m bert_pipeline.run_bert --runs 3 --no-lemmatize

# Try different models or settings
uv run python -m bert_pipeline.run_bert --model-name bert-base-uncased --epochs 5
```

### BERT Configuration

Default settings for BERT training:
- **Model**: DistilBERT (distilbert-base-uncased)
- **Loss**: Focal Loss with balanced class weights
- **Epochs**: 3 (configurable)
- **Batch Size**: 16
- **Learning Rate**: 2e-5

### Available Flags

```bash
--model-name      # Try different BERT models (bert-base-uncased, roberta-base)
--no-lemmatize    # Use raw text instead of lemmatized
--learning-rate   # Adjust learning rate (default: 2e-5)
--epochs          # Number of training epochs
--batch-size      # Batch size
--class-weights   # balanced, inverse, or none
--no-focal-loss   # Use cross-entropy instead of focal loss
--runs            # Multiple runs with different seeds
--compare         # Compare predefined configurations
--save-model      # Save the trained model
```

## 🛠️ Installation & Usage

This project uses `uv` (https://github.com/astral-sh/uv) for fast and reliable dependency management via `pyproject.toml`.

1. Clone the repository.
2. Ensure you have `uv` installed (`curl -LsSf https://astral.sh/uv/install.sh | sh` or via your preferred package manager).
3. Sync the project dependencies and create the virtual environment:
   ```bash
   uv sync
   ```
4. Run the evaluation loop to test linear models:
   ```bash
   uv run main.py
   ```
5. Train BERT on augmented data:
   ```bash
   uv run python -m bert_pipeline.run_bert
   ```

## 📁 Repository Structure

**Core Files:**
- `main.py` - Primary execution script and 5-run evaluation loop for linear models
- `data_reader.py` - Parses TREC formatting and handles lemmatization
- `sklearn_models.py` / `perceptron.py` - Core model implementations
- `evaluate.py` - Accuracy and F1-score metrics with per-class breakdown

**BERT Pipeline ([bert_pipeline/](bert_pipeline/)):**
- `run_bert.py` - Standalone BERT training and evaluation script
- `create_augmented_dataset.py` - Data augmentation orchestration
- `synthetic_generator.py` - LLM-based synthetic question generation
- `back_translation_augment.py` - Paraphrase augmentation via translation
- `data_balancer.py` - Strategic downsampling of majority classes
- `bert_classifier.py` - BERT model implementation with focal loss
- `utils.py` - Shared utilities and data loading

**Analysis & Visualization:**
- `feature_importance.py` - Extracts and visualizes SVM coefficients
- `error_analysis.py` - Analyzes model errors and zero-recall classes
- `confusion_plot.py` - Confusion matrix visualization
- `label_count.py` - Class imbalance visualization

**Hierarchical Models:**
- `hierarchical_svm.py` - Hierarchical architecture with LinearSVC base
- `hierarchical_perceptron.py` - Hierarchical architecture with AveragedPerceptron base
- `hybrid_kmeans.py` - K-means clustering within coarse classes

## 🚧 Conclusions

### Why Data Augmentation Was The Breakthrough

The **794:1 class imbalance** was the fundamental barrier preventing all models — especially neural networks — from learning effectively.

**On Original Data (794:1 imbalance):**
- BERT: 32-48% accuracy (insufficient data for fine-tuning)
- Linear models: 64-67% accuracy (trigger word matching)
- MLP: 58% accuracy (overfitting on sparse features)

**On Augmented Data (7.6:1 imbalance):**
- BERT: **87.40%** accuracy (+40-55% improvement) ✅
- Linear models: 70-75% accuracy (+5-10% improvement)
- MLP: 68% accuracy (+10% improvement)

**Key takeaway**: The 794:1 imbalance was the primary bottleneck, not the model architecture. With balanced data, semantic understanding (BERT) significantly outperforms linear approaches, but even linear models benefit substantially from addressing class imbalance.

### Per-Class Performance (Best Model)

**Perfect Performers (F1 = 1.0)**:
- ABBR:abb, ENTY:instru, ENTY:lang, ENTY:sport
- DESC:manner, NUM:date, NUM:period, NUM:speed

**Strong Performers (F1 > 0.90)**:
- HUM:ind (0.98), LOC:city (0.88), ENTY:animal (0.87)
- ENTY:currency (0.86), NUM:count (0.82), LOC:other (0.90)

**Still Challenging (Zero/Low Recall)**:
- ENTY:other, HUM:title - Generic trigger words remain problematic

### Original Dataset Challenges

The original TREC dataset's severe imbalance caused fundamental issues:

| Issue | Impact |
|-------|--------|
| 794:1 imbalance | Neural networks couldn't learn |
| 6 classes with ≤3 samples | Zero recall even after augmentation |
| Generic trigger words ("what") | Cross-class confusion |

### Future Directions

1. **Further Augmentation**: Generate more samples for remaining zero-recall classes
2. **Ensemble Methods**: Combine BERT with linear models for hybrid approach
3. **What-is Disambiguation**: Specialized handling for ambiguous "What is..." questions
4. **Alternative Architectures**: Experiment with larger BERT variants or different neural architectures

---

**Key Takeaway**: Data augmentation that addresses severe class imbalance enables neural networks to significantly outperform linear approaches on this task. The 794:1 imbalance was the primary bottleneck, not the task itself.
