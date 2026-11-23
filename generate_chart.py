import matplotlib.pyplot as plt
import numpy as np

# Data from the table
methods = [
    "Majority Baseline",
    "Random Baseline",
    "Std. Perceptron (Unigram)",
    "Std. Perceptron (Bigram)",
    "Avg. Perceptron (Unigram)",
    "Avg. Perceptron (Bigram)"
]

dev_means = [0.0064, 0.0161, 0.5978, 0.5297, 0.6469, 0.5270]
dev_stds = [0.0000, 0.0027, 0.0416, 0.0299, 0.0341, 0.0122]

test_means = [0.0047, 0.0152, 0.5950, 0.5071, 0.6340, 0.5300]
test_stds = [0.0000, 0.0075, 0.0199, 0.0298, 0.0221, 0.0123]

x = np.arange(len(methods))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars with error bars
rects1 = ax.bar(x - width/2, dev_means, width, yerr=dev_stds, label='Dev Score',
                color='#2b5c8f', capsize=5, edgecolor='black', alpha=0.9)
rects2 = ax.bar(x + width/2, test_means, width, yerr=test_stds, label='Test Score',
                color='#d95f02', capsize=5, edgecolor='black', alpha=0.9)

# Labels and styling
ax.set_ylabel(r'Scores (Mean $\pm$ Std)', fontsize=12)
ax.set_title('Model Performance Comparison against Baselines', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=25, ha="right", fontsize=10)
ax.legend(fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.5)

ax.bar_label(rects1, padding=15, fmt='%.4f%%')
ax.bar_label(rects2, padding=15, fmt='%.4f%%')

plt.tight_layout()
plt.show()
