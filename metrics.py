import numpy as np
from sklearn.metrics import ndcg_score
from scipy.stats import spearmanr

def calculate_metrics(y_true, y_pred, top_k=25):
    y_true = np.asarray(y_true).reshape(1, -1)
    y_pred = np.asarray(y_pred).reshape(1, -1)

    ndcg = ndcg_score(y_true, y_pred, k=top_k)
    spearman = spearmanr(y_true.ravel(), y_pred.ravel()).correlation
    return ndcg, spearman