from typing import Dict, Tuple

import pandas as pd
from sklearn.metrics import accuracy_score, hamming_loss, jaccard_score, precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer

from .labels import ALL_ERROR_LABELS


def compute_metrics(result_df: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame], Dict[str, float]]:
    output_status = result_df.get("_output_status")
    if output_status is None:
        output_status = pd.Series(["ok"] * len(result_df))

    error_count = int((output_status == "error").sum())
    unknown_count = int((output_status == "unknown").sum())

    y_true_bin = result_df["_gold_is_reasonable"].astype(int).tolist()
    y_pred_bin = result_df["_pred_is_reasonable"].astype(int).tolist()
    bin_acc = accuracy_score(y_true_bin, y_pred_bin)
    bin_p, bin_r, bin_f1, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average="binary", zero_division=0
    )

    gold_label_sets = result_df["_gold_labels_obj"].tolist()
    pred_label_sets = result_df["_pred_labels_obj"].tolist()
    mlb = MultiLabelBinarizer(classes=ALL_ERROR_LABELS)
    y_true = mlb.fit_transform(gold_label_sets)
    y_pred = mlb.transform(pred_label_sets)

    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    subset_acc = (y_true == y_pred).all(axis=1).mean()
    ham_loss = hamming_loss(y_true, y_pred)
    jaccard_samples = jaccard_score(y_true, y_pred, average="samples", zero_division=0)

    gold_edges_all = []
    pred_edges_all = []
    for _, row in result_df.iterrows():
        pid = str(row["prescription_id"])
        gold_edge_set = set((pid, a, b, label) for a, b, label in row["_gold_relation_edges_obj"])
        pred_edge_set = set((pid, a, b, label) for a, b, label in row["_pred_relation_edges_obj"])
        gold_edges_all.extend(list(gold_edge_set))
        pred_edges_all.extend(list(pred_edge_set))

    gold_edge_set_all = set(gold_edges_all)
    pred_edge_set_all = set(pred_edges_all)
    tp = len(gold_edge_set_all & pred_edge_set_all)
    fp = len(pred_edge_set_all - gold_edge_set_all)
    fn = len(gold_edge_set_all - pred_edge_set_all)
    edge_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    edge_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    edge_f1 = 2 * edge_precision * edge_recall / (edge_precision + edge_recall) if (edge_precision + edge_recall) > 0 else 0

    metric_sheets = {
        "binary": pd.DataFrame([
            {"task": "prescription_binary", "metric": "Accuracy", "value": bin_acc},
            {"task": "prescription_binary", "metric": "Precision", "value": bin_p},
            {"task": "prescription_binary", "metric": "Recall", "value": bin_r},
            {"task": "prescription_binary", "metric": "F1", "value": bin_f1},
            {"task": "prescription_binary", "metric": "Error Count", "value": error_count},
            {"task": "prescription_binary", "metric": "Unknown Count", "value": unknown_count},
        ]),
        "multilabel": pd.DataFrame([
            {"task": "prescription_multilabel", "metric": "Micro Precision", "value": micro_p},
            {"task": "prescription_multilabel", "metric": "Micro Recall", "value": micro_r},
            {"task": "prescription_multilabel", "metric": "Micro F1", "value": micro_f1},
            {"task": "prescription_multilabel", "metric": "Macro Precision", "value": macro_p},
            {"task": "prescription_multilabel", "metric": "Macro Recall", "value": macro_r},
            {"task": "prescription_multilabel", "metric": "Macro F1", "value": macro_f1},
            {"task": "prescription_multilabel", "metric": "Weighted Precision", "value": weighted_p},
            {"task": "prescription_multilabel", "metric": "Weighted Recall", "value": weighted_r},
            {"task": "prescription_multilabel", "metric": "Weighted F1", "value": weighted_f1},
            {"task": "prescription_multilabel", "metric": "Hamming Loss", "value": ham_loss},
            {"task": "prescription_multilabel", "metric": "Jaccard Score(samples)", "value": jaccard_samples},
            {"task": "prescription_multilabel", "metric": "Subset Accuracy", "value": subset_acc},
        ]),
        "edge": pd.DataFrame([
            {"task": "relation_edge", "metric": "TP", "value": tp},
            {"task": "relation_edge", "metric": "FP", "value": fp},
            {"task": "relation_edge", "metric": "FN", "value": fn},
            {"task": "relation_edge", "metric": "Precision", "value": edge_precision},
            {"task": "relation_edge", "metric": "Recall", "value": edge_recall},
            {"task": "relation_edge", "metric": "F1", "value": edge_f1},
        ]),
    }

    summary = {
        "binary_accuracy": bin_acc,
        "binary_f1": bin_f1,
        "multilabel_micro_f1": micro_f1,
        "multilabel_macro_f1": macro_f1,
        "edge_f1": edge_f1,
        "error_count": error_count,
        "unknown_count": unknown_count,
    }
    return metric_sheets, summary
