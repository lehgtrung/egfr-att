
import numpy as np
import sklearn.metrics as metrics
THRESH=0.5

def auc(y_true, y_scores):
    y_true = y_true.cpu().detach().numpy()
    y_scores = y_scores.cpu().detach().numpy()
    return metrics.roc_auc_score(y_true, y_scores)


def auc_threshold(y_true, y_scores):
    y_true = y_true.cpu().detach().numpy()
    y_scores = y_scores.cpu().detach().numpy()
    fpr, tpr, threshold = metrics.roc_curve(y_true, y_scores)
    return metrics.auc(fpr, tpr)


def get_score_obj(y_true, y_scores, thresh=THRESH):
    y_true = y_true.cpu().detach().numpy()
    y_scores = (y_scores.cpu().detach().numpy() + thresh).astype(np.int16)
    return metrics.classification_report(y_true, y_scores, output_dict=True)


def f1(y_true, y_scores):
    score_obj = get_score_obj(y_true, y_scores)
    return score_obj['weighted avg']['f1-score']


def class1_precision(y_true, y_scores):
    score_obj = get_score_obj(y_true, y_scores)
    return score_obj['1.0']['precision']


def class1_recall(y_true, y_scores):
    score_obj = get_score_obj(y_true, y_scores)
    return score_obj['1.0']['recall']

# Metrics for benchmark

def sensitivity(y_true, y_scores, thresh=THRESH):
    y_true = y_true.cpu().detach().numpy()
    y_scores = (y_scores.cpu().detach().numpy() + thresh).astype(np.int16)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_scores).ravel()
    return tp / (tp + fn)


def specificity(y_true, y_scores, thresh=THRESH):
    y_true = y_true.cpu().detach().numpy()
    y_scores = (y_scores.cpu().detach().numpy() + thresh).astype(np.int16)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_scores).ravel()
    return tn / (tn + fp)


def accuracy(y_true, y_scores, thresh=THRESH):
    y_true = y_true.cpu().detach().numpy()
    y_scores = (y_scores.cpu().detach().numpy() + thresh).astype(np.int16)
    return metrics.accuracy_score(y_true, y_scores)



def mcc(y_true, y_scores, thresh=THRESH):
    y_true = y_true.cpu().detach().numpy()
    y_scores = (y_scores.cpu().detach().numpy() + thresh).astype(np.int16)
    return metrics.matthews_corrcoef(y_true, y_scores)




