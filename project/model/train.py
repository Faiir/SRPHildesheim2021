import time
from collections import defaultdict
from tqdm import tqdm
import torch
from torchsummary import summary
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from ..data.datahandler_for_array import get_ood_dataloader
from ..helpers.early_stopping import EarlyStopping
import gc
import torch.backends.cudnn as cudnn




def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


def create_lr_sheduler(optimizer, epochs, pert_loader, learning_rate):
    """create_lr_sheduler [Creates lr scheduler with cosine shedule]

    Args:
        optimizer ([torch.optim]): [description]
        epochs ([int]): [max epochs]
        pert_loader ([dataloader]): [description]
        learning_rate ([float]): [initial learning rate]

    Returns:
        [torch.lr_scheduler]: [description]
    """
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            epochs * len(pert_loader),
            1,  # since lr_lambda computes multiplicative factor
            1e-6 / learning_rate,
        ),
    )


def verbosity(message, verbose, epoch):
    if verbose == 1:
        if epoch % 10 == 0:
            print(message)
    elif verbose == 2:
        print(message)
    return None


def train(net, train_loader, optimizer, criterion, device, epochs=5, **kwargs):
    """train [main training function of the project]

    [extended_summary]

    Args:
        net ([torch.nn.Module]): [Neural network to train]
        train_loader ([torch.Dataloader]): [dataloader with the training data]
        optimizer ([torch.optim]): [optimizer for the network]
        criterion ([Loss function]): [Pytorch loss function]
        device ([str]): [device to train on cpu/cuda]
        epochs (int, optional): [epochs to run]. Defaults to 5.
        **kwargs (verbose and validation dataloader)
    Returns:
        [tupel(trained network, train_loss )]:
    """
    verbose = kwargs.get("verbose", 1)
    val_dataloader = kwargs.get("val_dataloader", None)
    cudnn.benchmark = True

    if verbose > 0:
        print("\nTraining with device :", device)
        print("Number of Training Samples : ", len(train_loader.dataset))
        if val_dataloader is not None:
            print("Number of Validation Samples : ", len(val_dataloader.dataset))
        print("Number of Epochs : ", epochs)

        if verbose > 1:
            summary(net, input_size=(3, 32, 32))

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    if kwargs.get("lr_sheduler", True):
        lr_sheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            "min",
            factor=0.1,
            patience=int(epochs * 0.05),
            min_lr=1e-7,
            verbose=True,
        )

    if val_dataloader is not None:
        validation = True
        if kwargs.get("patience", None) is None:
            print(f'INFO ------ Early Stopping Patience not specified using {int(epochs * 0.1)}')
        patience = kwargs.get("patience", int(epochs * 0.1))
        early_stopping = EarlyStopping(patience, verbose=True, delta=1e-6)
    else:
        validation = False

    for epoch in tqdm(range(1, epochs + 1)):
        if verbose > 0:
            print(f"\nEpoch: {epoch}")

        train_loss = 0
        train_acc = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            if len(data) > 1:
                net.train()
                data, target = data.to(device).float(), target.to(device).long()

                optimizer.zero_grad(set_to_none=True)
                yhat = net(data).to(device)
                loss = criterion(yhat, target)
                train_loss += loss.item()
                train_acc += torch.sum(torch.argmax(yhat, dim=1) == target).item()

                loss.backward()
                optimizer.step()
            else:
                pass

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader.dataset)

        if epoch % 1 == 0:
            if validation:
                val_loss = 0
                val_acc = 0
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
                        val_acc += torch.sum(
                            torch.argmax(voutput, dim=1) == vtarget
                        ).item()

                avg_val_loss = val_loss / len(val_dataloader)
                avg_val_acc = val_acc / len(val_dataloader.dataset)

                early_stopping(avg_val_loss, net)
                if kwargs.get("lr_sheduler", True):
                    lr_sheduler.step(avg_val_loss)

                verbosity(
                    f"Val_loss: {avg_val_loss:.4f} Val_acc : {100*avg_val_acc:.2f}",
                    verbose,
                    epoch,
                )

                if early_stopping.early_stop:
                    print(
                        f"Early stopping epoch {epoch} , avg train_loss {avg_train_loss}, avg val loss {avg_val_loss}"
                    )
                    break

        verbosity(
            f"Train_loss: {avg_train_loss:.4f} Train_acc : {100*avg_train_acc:.2f}",
            verbose,
            epoch,
        )

    return net, avg_train_loss


def test(model, criterion, test_dataloader, device, verbose=0):
    """test [Function to measure performance on the test set]


    Args:
        model ([type]): [description]
        criterion ([type]): [description]
        test_dataloader ([type]): [description]
        device ([type]): [description]

    Returns:
        [float]: [avg. test loss]
    """
    test_loss = 0
    model.eval()
    for (t_data, t_target) in test_dataloader:
        
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



def pertube_image(pool_loader, val_loader, trained_net):
    gs = []
    hs = []
    pert_preds = []
    targets = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    epsi_list = [0.0025, 0.005, 0.01, 0.02, 0.04, 0.08]
    best_eps = 0
    scores = []
    trained_net.eval()
    for eps in tqdm(epsi_list):
        preds = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            trained_net.zero_grad(set_to_none=True)
            backward_tensor = torch.ones((data.size(0), 1)).float().to(device)
            data, target = data.to(device).float(), target.to(device).long()
            data.requires_grad = True
            output = trained_net(data, apply_softmax=True)
            pred, _ = output.max(dim=-1, keepdim=True)

            pred.backward(backward_tensor)
            pert_imgage = fgsm_attack(data, epsilon=eps, data_grad=data.grad.data)
            del data, output, target
            gc.collect()
    
            yhat = trained_net(pert_imgage, apply_softmax=True)
            pred = torch.max(yhat, dim=-1, keepdim=False, out=None).values
            preds += torch.sum(pred)
            del pred, yhat, pert_imgage
            gc.collect()
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
        output = trained_net(data, apply_softmax=True)
        pred, _ = output.max(dim=-1, keepdim=True)

        pred.backward(backward_tensor)
        pert_imgs.append(
            fgsm_attack(data, epsilon=eps, data_grad=data.grad.data).to("cpu")
        )
        targets.append(target.to("cpu").numpy().astype(np.float16))
        del data, output, target
        gc.collect()
    torch.cuda.empty_cache()
    trained_net.zero_grad(set_to_none=True)
    with torch.no_grad():
        for p_img in pert_imgs:
            pert_pred, g, h = trained_net(p_img.to(device), get_test_model=True, apply_softmax=True)
            gs.append(g.detach().to("cpu").numpy().astype(np.float16))
            hs.append(h.detach().to("cpu").numpy().astype(np.float16))
            pert_preds.append(pert_pred.detach().to("cpu").numpy())
            p_img.detach().to("cpu").numpy().astype(np.float16)
    del pert_imgs
    
    return pert_preds, gs, hs, targets


def get_density_vals(pool_loader, val_loader, trained_net, do_pertubed_images):
    """get_density_vals [Model to measure the density of the pool data to create distribution plots]

    [extended_summary]

    Args:
        pool_loader ([Dataloader]): [description]
        val_loader ([Dataloader]): [description]
        trained_net ([nn-Module]): [description]
        do_pertubed_images ([bool]): [description]

    Returns:
        [tupel(pert_preds, gs, hs, targets)]: [predictions for the pool data, coressponding g / h values, actual targets]
    """
    if do_pertubed_images:
        return pertube_image(pool_loader, val_loader, trained_net)
    else:
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(pool_loader):
                data = data.to(device).float()
                pert_pred, g, h = trained_net(data, get_test_model=True, apply_softmax=True)
                gs.append(g.detach().to("cpu").numpy().astype(np.float16))
                hs.append(h.detach().to("cpu").numpy().astype(np.float16))
                pert_preds.append(pert_pred.detach().to("cpu").numpy())
                targets.append(target.to("cpu").numpy().astype(np.float16))

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
