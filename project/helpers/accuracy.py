
import numpy as np


def accuracy(true_labels,predictions):
    predicted_labels = np.argmax(predictions,axis=1).flatten()
    return 100*np.mean(predicted_labels==(true_labels.flatten()))
