from sklearn.manifold import TSNE
import numpy as np
import seaborn
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import torch


def _transform(x):
    if x > 0:
        return 1
    else:
        return -1


def density_plot(pert_preds, gs, hs, targets, writer, oracle_step):
    """density_plot [plots the density of the predicted distribution via entropy and g values ]

    [extended_summary]

    Args:
        pert_preds ([array]): [description]
        gs ([array]): [description]
        hs ([arry]): [description]
        targets ([array]): [description]
        writer ([torch.SummaryWriter]): [description]
        oracle_step ([current AL iteration]): [description]
    """
    targets = np.concatenate(targets, axis=0)
    pert_preds = np.concatenate(pert_preds, axis=0)
    gs = np.concatenate(gs, axis=0)
    hs = np.concatenate(hs, axis=0)

    source = np.array([_transform(xi) for xi in np.array(targets)])
    entropies = -np.sum(pert_preds * np.log(pert_preds), axis=1)
    df_perturbed = pd.DataFrame(
        np.concatenate(
            [
                targets[..., np.newaxis],
                entropies[..., np.newaxis],
                source[..., np.newaxis],
                gs,
                np.max(hs, axis=1, keepdims=True),
            ],
            axis=1,
        ),
        columns=["labels", "entropies", "source", "g_s", "h_s"],
    )
    map_labels = {-1: "OoD"}
    map_labels.update({ii: f"In_dist{ii}" for ii in range(10)})
    source_labels = {-1: "OoD", 1: "InDist"}

    df_perturbed["source_names"] = df_perturbed.source.astype(int).map(source_labels)
    df_perturbed["g_s*e"] = df_perturbed["g_s"] * df_perturbed["entropies"]
    # fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 18))
    try:
        plot = seaborn.kdeplot(
            data=df_perturbed,
            x="g_s",
            y="entropies",
            hue="source_names",
            fill=False,
            ax=ax1,
        )
    except:
        print("Can't produce KDE, working on scatter plot instead")
        plot = seaborn.scatterplot(
            data=df_perturbed,
            x="g_s",
            y="entropies",
            hue="source_names",
            alpha=0.5,
            ax=ax1,
        )
    try:
        plot = seaborn.kdeplot(
            data=df_perturbed,
            x="g_s",
            y="g_s*e",
            hue="source_names",
            fill=False,
            ax=ax2,
        )
    except IndexError:
        print("Can't produce KDE, working on scatter plot instead")
        plot = seaborn.scatterplot(
            data=df_perturbed,
            x="g_s",
            y="g_s*e",
            hue="source_names",
            alpha=0.5,
            ax=ax2,
        )
    seaborn.histplot(
        data=df_perturbed,
        x="g_s",
        hue="source_names",
        ax=ax3,
    )
    writer.add_figure(tag=f"density_oracle_step_{oracle_step}", figure=fig)


def map_unlabelled(x):
    x[x >= 10] = 1
    return x


def get_tsne_plot(data_manager, dataset, net, device, plot_ood=True):

    labelled_data, labelled_labels = data_manager.get_train_data()

    unlabelled_data, unlabelled_labels = data_manager.get_unlabelled_pool_data()

    # Merge labelled with unlabelled data:
    unlabelled_labels[unlabelled_labels >= 0] += 10
    merged_data = torch.Tensor(np.concatenate((labelled_data, unlabelled_data))).to(
        device
    )
    labels = np.concatenate((labelled_labels, unlabelled_labels))
    source = np.concatenate(
        [np.zeros_like(labelled_labels), map_unlabelled(unlabelled_labels)]
    )

    # -1: OOD
    # 0: already labelled
    # 1: unlabelled

    with torch.no_grad():
        predictions = net(merged_data.float())
        tsne_embeddings = TSNE(n_components=2, perplexity=15).fit_transform(
            predictions.cpu()
        )

    df = pd.DataFrame(
        np.concatenate([labels[..., np.newaxis], source[..., np.newaxis]], axis=1),
        columns=["labels", "source"],
    )

    map_labels = {-1: f"{dataset}"}
    map_labels.update({ii: f"Trained_{ii}" for ii in range(10)})
    map_labels.update({ii: f"Untrained_{ii-10}" for ii in range(10, 20)})

    source_labels = {-1: "OoD", 0: "Trained_ID", 1: "Untrained_ID"}
    df["source_names"] = df.source.astype(int).map(source_labels)
    df["label_names"] = df.labels.astype(int).map(map_labels)
    df["first"] = tsne_embeddings[:, 0]
    df["second"] = tsne_embeddings[:, 1]

    colours = ["#000000"]
    colours_10 = [
        "#E53935",
        "#F4511E",
        "#283593",
        "#03A9F4",
        "#00ACC1",
        "#689F38",
        "#FBC02D",
        "#FB8C00",
        "#7B1FA2",
        "#757575",
    ]

    colours_20 = [
        "#F5B7B1",
        "#EDBB99",
        "#7FB3D5",
        "#D6EAF8",
        "#80DEEA",
        "#C5E1A5",
        "#FFF59D",
        "#FAD7A0",
        "#D2B4DE",
        "#E5E7E9",
    ]

    hue_order = [f"{dataset}"]
    hue_order_10 = [f"Trained_{ii}" for ii in range(10)]
    hue_order_20 = [f"Untrained_{ii-10}" for ii in range(10, 20)]

    df["first"] = tsne_embeddings[:, 0]
    df["second"] = tsne_embeddings[:, 1]

    sns.set(rc={"axes.facecolor": "#F9F9F9"})

    s = 15
    alpha = 0.8

    fig, ax = plt.subplots(1, 1, figsize=(18, 18))
    sns.scatterplot(
        x="first",
        y="second",
        data=df[df["source"] == 1],
        hue="label_names",
        hue_order=hue_order_20,
        palette=colours_20,
        legend="full",
        s=s,
        alpha=alpha,
        ax=ax,
    )
    sns.scatterplot(
        x="first",
        y="second",
        data=df[df["source"] == 0],
        hue="label_names",
        hue_order=hue_order_10,
        palette=colours_10,
        legend="full",
        s=s,
        ax=ax,
    )
    sns.scatterplot(
        x="first",
        y="second",
        data=df[df["source"] == -1],
        hue="label_names",
        hue_order=hue_order,
        palette=colours,
        legend="full",
        s=s,
        alpha=alpha,
        ax=ax,
    )

    ax.set_xlabel("First")
    ax.set_ylabel("Second")
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    return fig