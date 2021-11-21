from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def AUROC(iD_Prob, source_labels, writer, oracle_step, normalize=False, plot_auc= True):
    """
    Computes the ROC score and plots the ROC_AUC curve. 
    Inputs:
        iD_Prob - Probability (or Score) of pool samples being iD
        source_labels - pool labels (1: iD, 0: OoD)
        writer - tensorboard writer
        oracle_step
        normalize - flag to use if Score is provided instead of Probability.
        plot_auc - flag to plot the curves
    """

    if normalize:
        scaler = MinMaxScaler()
        perdictions = scaler.fit_transform(iD_Prob)
    
    score = roc_auc_score(perdictions, source_labels)
    if score<0.5:
        print('INFO ----- ROC function requires iD_Prob, not OoD_Prob')
        return AUROC(-iD_Prob, source_labels, writer, oracle_step, normalize)
   
    if plot_auc:
        fpr, tpr, _ = roc_curve(perdictions, source_labels)
        fig = plt.figure()
        lw = 1
        plt.plot(
            fpr,
            tpr,
            color="red",
            lw=lw,
            label="ROC curve (area = %0.2f)" % score,
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC")
        plt.legend(loc="lower right")
        writer.add_figure(tag=f"AUROC_{oracle_step}", figure=fig)

    return score



