import time
from collections import defaultdict
from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F
from ..data.datahandler_for_array import get_ood_dataloader
from ..helpers.early_stopping import EarlyStopping


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


def create_lr_sheduler(optimizer, epochs, pert_loader, learning_rate):

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            epochs * len(pert_loader),
            1,  # since lr_lambda computes multiplicative factor
            1e-6 / learning_rate,
        ),
    )


def train(
    net, train_loader, optimizer, criterion, device, epochs=5, verbose=1, **kwargs
):
    if verbose > 0:
        print("training with device:", device)
    validation = False
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    if kwargs.get("lr_sheduler", True):
        lr_sheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            "min",
            factor=0.3,
            patience=int(epochs * 0.05),
            min_lr=1e-7,
            verbose=True,
        )

    if kwargs.get("do_validation", False):
        validation = True
        val_dataloader = kwargs.get("val_dataloader")
        patience = kwargs.get("patience", 10)
        early_stopping = EarlyStopping(patience, verbose=True, delta=1e-6)

    for epoch in tqdm(range(0, epochs)):
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):

            net.train()
            data, target = data.to(device).float(), target.to(device).long()

            optimizer.zero_grad(set_to_none=True)
            yhat = net(data).to(device)
            loss = criterion(yhat, target)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_train_loss = train_loss / len(train_loader)

        if epoch % 1 == 0:
            if validation:
                val_loss = 0
                net.eval()  # prep model for evaluation
                with torch.no_grad():
                    for vdata, vtarget in val_dataloader:
                        vdata, vtarget = (
                            vdata.to(device).float(),
                            vtarget.to(device).long(),
                        )
                        voutput = net(vdata)
                        vloss = criterion(voutput, vtarget)
                        val_loss += vloss.item()

                avg_val_loss = val_loss / len(val_dataloader)
                early_stopping(avg_val_loss, net)
                if kwargs.get("lr_sheduler", True):
                    lr_sheduler.step(avg_val_loss)
                print(f"avg val loss {avg_val_loss} epoch {epoch}")
                if early_stopping.early_stop:
                    print(
                        f"Early stopping epoch {epoch} , avg train_loss {avg_train_loss}, avg val loss {avg_val_loss}"
                    )

                    break

        if verbose == 1:
            if epoch % 10 == 0:
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
    trained_net.eval()
    for eps in tqdm(epsi_list):
        preds = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            trained_net.zero_grad(set_to_none=True)
            data, target = data.to(device).float(), target.to(device).long()
            data.requires_grad = True
            yhat = trained_net(data)
            pred = torch.max(yhat, dim=-1, keepdim=False, out=None).values

            preds += torch.sum(pred)

            del data, target, pred
        scores.append(preds.detach().cpu().numpy())
    torch.cuda.empty_cache()
    trained_net.zero_grad(set_to_none=True)
    eps = epsi_list[np.argmax(scores)]
    del scores
    pert_imgs = []
    targets = []
    for batch_idx, (data, target) in enumerate(pool_loader):
        trained_net.zero_grad(set_to_none=True)
        backward_tensor = torch.ones((data.size(0), 1)).float().to(device)
        data, target = data.to(device).float(), target.to(device).long()
        data.requires_grad = True
        output = trained_net(data)
        pred, _ = output.max(dim=-1, keepdim=True)

        pred.backward(backward_tensor)
        pert_imgs.append(
            fgsm_attack(data, epsilon=eps, data_grad=data.grad.data).to("cpu")
        )
        targets.append(target.to("cpu").numpy().astype(np.float16))
        del data, output, target
    torch.cuda.empty_cache()
    trained_net.zero_grad(set_to_none=True)
    gs = []
    hs = []
    pert_preds = []
    with torch.no_grad():
        for p_img in pert_imgs:
            pert_pred, g, h = trained_net(p_img.to(device), get_test_model=True)
            gs.append(g.detach().to("cpu").numpy().astype(np.float16))
            hs.append(h.detach().to("cpu").numpy().astype(np.float16))
            pert_preds.append(pert_pred.detach().to("cpu").numpy())
            p_img.detach().to("cpu").numpy().astype(np.float16)
    del pert_imgs
    return pert_preds, gs, hs, targets


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def train_g(net, optimizer, datamanager, epochs=10):
    """train_g [summary]

    [Outlier exposure for g-head - https://arxiv.org/pdf/1812.04606.pdf]


    """
    net.train()  # enter train mode
    loss_avg = 0.0

    train_loader_in, train_loader_out = get_ood_dataloader(datamanager)

    for epoch in range(epochs):
        train_in, target_in = next(iter(train_loader_in))

    _train_out_list = []
    _target_out_list = []

    for i in range(2):
        _train_out, _target_out = next(iter(train_loader_out))
        _train_out_list.append(_train_out)
        _target_out_list.append(_target_out)

    train_out = torch.cat((_train_out_list[0], _train_out_list[1]), 0)
    target_out = torch.cat((_target_out_list[0], _target_out_list[1]), 0)

    # for in_set, out_set in zip(train_loader_in, train_loader_out):
    data = torch.cat((train_in, train_out), 0)
    target = torch.cat((target_in, target_out), 0)

    data, target = data.cuda(), target.cuda()

    optimizer.zero_grad(set_to_none=True)
    x = net(data, train_g=True)

    loss = F.binary_cross_entropy(x, target)
    loss.backward()
    loss_avg += loss

    optimizer.step()

    # loss = F.cross_entropy(x[: len(in_set[0])], target)
    # # cross-entropy from softmax distribution to uniform distribution
    # loss += (
    #     0.5
    #     * -(
    #         x[len(in_set[0]) :].mean(1)
    #         - torch.logsumexp(x[len(in_set[0]) :], dim=1)
    #     ).mean()
    # )

    print(f"outlier exposure average loss: {loss_avg/epochs}")
    return net


def pretrain_self_sup(net, loader, optimizer):

    pass
