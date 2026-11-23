import matplotlib.pyplot as plt
import data_reader 
from collections import Counter

def main():
    # Load and count
    full_lem = data_reader.read("data/train_5500.label", lemmatize=True)
    train_data = full_lem[:4500] 
    labels = [item.label for item in train_data]
    label_counts = Counter(labels)

    # Sort by count
    sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=False)
    categories = [item[0] for item in sorted_labels]
    counts = [item[1] for item in sorted_labels]

    plt.figure(figsize=(10, 12))
    plt.barh(categories, counts, color='steelblue')
    
    plt.title("TREC Dataset: Severe Class Imbalance", fontsize=16, pad=20)
    plt.xlabel("Number of Training Examples", fontsize=12)
    plt.ylabel("Question Category", fontsize=12)
    
    for index, value in enumerate(counts):
            plt.text(value + 5, index, str(value), va='center', fontsize=8)

    plt.margins(y=0.01)
    plt.tight_layout()
    plt.savefig("class_imbalance_chart.png", dpi=300)
    print("Saved!")

if __name__ == "__main__":
    main()