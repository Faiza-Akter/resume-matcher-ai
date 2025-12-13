from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def compute_metrics(y_true, y_pred):
    """
    Compute all important AI evaluation metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_true, y_pred, zero_division=0
        ),
    }


def metrics_to_dict(metrics):
    """
    Convert metrics for saving into JSON.
    """
    return {
        "accuracy": round(metrics["accuracy"] * 100, 2),
        "precision": round(metrics["precision"] * 100, 2),
        "recall": round(metrics["recall"] * 100, 2),
        "f1_score": round(metrics["f1_score"] * 100, 2),
        "confusion_matrix": metrics["confusion_matrix"],
    }
