
import numpy as np
from sklearn.metrics import f1_score

def f1(true_labels,predictions, average = "macro"):
    predicted_labels = np.argmax(predictions,axis=1).flatten()
    f1score = f1_score(true_labels, predicted_labels, average = average)
    return f1score
