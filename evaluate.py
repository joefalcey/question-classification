from typing import List, Dict, Tuple
from collections import defaultdict
import data_reader
import majority_baseline


def evaluate(true_labels: List[str], predicted_labels: List[str], verbose: bool = True):
    """
    Evaluate predictions and return macro F1 score.
    If verbose=True, prints per-class metrics.
    """
    total_instances = len(true_labels)
    correct_predictions = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred)
    accuracy = correct_predictions / total_instances

    if verbose:
        print(f"Overall Accuracy: {accuracy * 100:.2f}%\n")

    classes = set(true_labels)
    if verbose:
        print(f"{'Class':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")

    tp_dict = defaultdict(int)
    fp_dict = defaultdict(int)
    fn_dict = defaultdict(int)
    support_dict = defaultdict(int)

    for true_l, pred_l in zip(true_labels, predicted_labels):
        support_dict[true_l] += 1
        if true_l == pred_l:
            tp_dict[true_l] += 1
        else:
            fp_dict[pred_l] += 1
            fn_dict[true_l] += 1

    total_f1 = 0.0
    global_tp = global_fp = global_fn = 0

    for c in sorted(classes):
        tp = tp_dict[c]
        fp = fp_dict[c]
        fn = fn_dict[c]
        support = support_dict[c]

        global_tp += tp
        global_fp += fp
        global_fn += fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        total_f1 += f1

        if verbose:
            print(f"{c:<12} | {precision:.4f}     | {recall:.4f}     | {f1:.4f}     | {support:<8}")

    macro_f1 = total_f1 / len(classes)

    micro_precision = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0.0
    micro_recall = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0.0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

    if verbose:
        print("-" * 65)
        print(f"{'MACRO AVG':<12} | {'':<10} | {'':<10} | {macro_f1:.4f}     | {total_instances:<8}")
        print(f"{'MICRO AVG':<12} | {micro_precision:.4f}     | {micro_recall:.4f}     | {micro_f1:.4f}     | {total_instances:<8}\n")

    return macro_f1


def get_detailed_metrics(true_labels: List[str], predicted_labels: List[str]) -> Dict[str, Tuple[float, float, float, int]]:
    """
    Return detailed per-class metrics without printing.
    Returns: Dict mapping class -> (precision, recall, f1, support)
    """
    classes = set(true_labels)

    tp_dict = defaultdict(int)
    fp_dict = defaultdict(int)
    fn_dict = defaultdict(int)
    support_dict = defaultdict(int)

    for true_l, pred_l in zip(true_labels, predicted_labels):
        support_dict[true_l] += 1
        if true_l == pred_l:
            tp_dict[true_l] += 1
        else:
            fp_dict[pred_l] += 1
            fn_dict[true_l] += 1

    metrics = {}
    total_f1 = 0.0

    for c in sorted(classes):
        tp = tp_dict[c]
        fp = fp_dict[c]
        fn = fn_dict[c]
        support = support_dict[c]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[c] = (precision, recall, f1, support)
        total_f1 += f1

    # Add macro and micro averages
    macro_f1 = total_f1 / len(classes)
    metrics["MACRO AVG"] = (0.0, 0.0, macro_f1, len(true_labels))

    # Calculate micro averages
    global_tp = sum(tp_dict.values())
    global_fp = sum(fp_dict.values())
    global_fn = sum(fn_dict.values())

    micro_precision = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0.0
    micro_recall = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0.0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

    metrics["MICRO AVG"] = (micro_precision, micro_recall, micro_f1, len(true_labels))

    return metrics


def print_detailed_report(true_labels: List[str], predicted_labels: List[str], model_name: str = "Model"):
    # Print a detailed classification report for a model
    print(f"\n{'='*75}")
    print(f"DETAILED REPORT: {model_name}")
    print(f"{'='*75}\n")

    metrics = get_detailed_metrics(true_labels, predicted_labels)

    print(f"{'Class':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 75)

    for class_name in sorted(metrics.keys()):
        precision, recall, f1, support = metrics[class_name]
        if class_name in ["MACRO AVG", "MICRO AVG"]:
            print("-" * 75)
            print(f"{class_name:<12} | {precision:.4f}     | {recall:.4f}     | {f1:.4f}     | {support:<8}")
        else:
            print(f"{class_name:<12} | {precision:.4f}     | {recall:.4f}     | {f1:.4f}     | {support:<8}")

    print()
