
import numpy as np
from sklearn.metrics import roc_auc_score


def auc(y_true, y_scores):
    y_true = y_true.cpu().detach().numpy()
    y_scores = y_scores.cpu().detach().numpy()
    return roc_auc_score(y_true, y_scores)

