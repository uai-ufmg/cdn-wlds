import numpy as np

def mean_reciprocal_rank(y_true, y_pred):
    reciprocal_ranks = []
    for true, pred in zip(y_true, y_pred):
        for i, p in enumerate(pred):
            if p == true:
                reciprocal_ranks.append(1 / (i + 1))
    return np.mean(reciprocal_ranks)

def accuracy_one_sequence(y_true, y_pred, k):
    for i, pred in enumerate(y_pred[:k]):
        if pred == y_true:
            return 1
    return 0

def accuracy_at_k(y_true, y_pred, k):
    return np.mean([accuracy_one_sequence(true, pred, k) for true, pred in zip(y_true, y_pred)])