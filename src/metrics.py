
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

def auc_score(y_true, y_prob):
    return float(roc_auc_score(y_true, y_prob))

def eer_score(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = (fpr[i] + fnr[i]) / 2.0
    return float(eer), float(thr[i])

def confusion(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)
