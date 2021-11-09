import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


def f1(true_labels, predictions, average="macro"):
    predicted_labels = np.argmax(predictions, axis=1).flatten()
    f1score = f1_score(true_labels, predicted_labels, average=average)
    return f1score


def accuracy(true_labels, predictions):
    predicted_labels = np.argmax(predictions, axis=1).flatten()
    return 100 * np.mean(predicted_labels == (true_labels.flatten()))


def auroc(data_manager, curr_iter, predictions):
    raise NotImplementedError
    pos_ex = data_manager.status_manager[
        data_manager.status_manager["status"] == curr_iter
    ]
    neg_ex = data_manager.status_manager[
        data_manager.status_manager["status"] == -curr_iter
    ]
    roc_score = ()
    return 3