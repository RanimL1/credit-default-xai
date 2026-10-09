import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)


def ks_stat(y_true, proba):
    fpr, tpr, _ = roc_curve(y_true, proba)
    return float(np.max(tpr - fpr))


def metriques(y_true, proba, seuil=0.5):
    pred = (np.asarray(proba) >= seuil).astype(int)
    auc = roc_auc_score(y_true, proba)
    return {
        "AUC-ROC": auc,
        "PR-AUC": average_precision_score(y_true, proba),
        "Précision": precision_score(y_true, pred, zero_division=0),
        "Rappel": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "Gini": 2 * auc - 1,
        "KS": ks_stat(y_true, proba),
    }


def tableau_comparatif(y_true, probas, seuil=0.5):
    return (pd.DataFrame({nom: metriques(y_true, p, seuil) for nom, p in probas.items()})
            .T.round(3).sort_values("AUC-ROC", ascending=False))
