
import numpy as np
from sklearn.metrics import roc_auc_score

def auroc(data_manager, curr_iter):
    print(data_manager.status_manager[data_manager.status_manager["status"] == curr_iter])
    # datamanager.status_manager seems to only save indist?
    return 3
