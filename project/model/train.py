import time
from collections import defaultdict
from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F


def train(
    net, train_loader, optimizer, criterion, device, do_fgsm=False, epochs=5, verbose=1
):
    if verbose > 0:
        print("training with device:", device)

    train_log = defaultdict(list)

    for epoch in tqdm(range(0, epochs)):
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):

            net.train()
            data, target = data.to(device).float(), target.to(device).long()

            optimizer.zero_grad()
            yhat = net(data).to(device)
            loss = criterion(yhat, target)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_train_loss = train_loss / len(train_loader)

        if verbose == 1:
            if epoch % (epochs // 10) == 0:
                print(" epoch: ", epoch, "current train_loss:", avg_train_loss)
        elif verbose == 2:
            print(" epoch: ", epoch, "current train_loss:", avg_train_loss)

    return net, avg_train_loss


def test(model, criterion, test_dataloader, device, verbose=0):
    test_loss = 0

    for (t_data, t_target) in test_dataloader:
        model.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        t_data, t_target = t_data.to(device).float(), t_target.to(device).long()

        with torch.no_grad():
            yhat = model(t_data)
            yhat.to(device).long()
            t_loss = criterion(yhat, t_target)
            test_loss += t_loss

    return test_loss.to("cpu").detach().numpy() / len(
        test_dataloader
    )  # return avg testloss


def get_density_vals(
    pool_loader,
    val_loader,
    trained_net,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epsi_list = [0.0025, 0.005, 0.01, 0.02, 0.04, 0.08]
    best_eps = 0
    scores = []
    for eps in tqdm(epsi_list):
        preds = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device).float(), target.to(device).long()
            data.requires_grad = True
            yhat = trained_net(data)
            pred = torch.max(yhat, dim=-1, keepdim=False, out=None).values

            preds += torch.sum(pred)
        scores.append(preds.detach().cpu().numpy())

    eps = epsi_list[np.argmax(scores)]
    pert_imgs = []

    for batch_idx, (data, target) in enumerate(pool_loader):
        backward_tensor = torch.ones((data.size(0), 1)).float().to(device)
        data, target = data.to(device).float(), target.to(device).long()
        data.requires_grad = True
        output = trained_net(data)
        pred, _ = output.max(dim=-1, keepdim=True)
        trained_net.zero_grad()
        pred.backward(backward_tensor)
        pert_imgs.append(fgsm_attack(data, epsilon=eps, data_grad=data.grad.data))

    gs = []
    hs = []
    pert_preds = []
    for p_img in pert_imgs:
        pert_pred, g, h = trained_net(p_img, get_test_model=True)
        gs.append(g)
        hs.append(h)
        pert_preds.append(pert_pred)

    return pert_imgs, pert_preds, gs, hs


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image