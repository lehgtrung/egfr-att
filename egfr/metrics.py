
import numpy as np
import sklearn.metrics as metrics


def auc(y_true, y_scores):
    y_true = y_true.cpu().detach().numpy()
    y_scores = y_scores.cpu().detach().numpy()
    return metrics.roc_auc_score(y_true, y_scores)


def auc_threshold(y_true, y_scores):
    y_true = y_true.cpu().detach().numpy()
    y_scores = y_scores.cpu().detach().numpy()
    fpr, tpr, threshold = metrics.roc_curve(y_true, y_scores)
    return metrics.auc(fpr, tpr)

