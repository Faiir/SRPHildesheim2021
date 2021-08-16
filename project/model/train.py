import time
from collections import defaultdict
from tqdm import tqdm
import torch


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
            if do_fgsm:
                data.requires_grad = True

            optimizer.zero_grad()
            yhat = net(data)
            yhat.to(device).long()
            loss = criterion(yhat, target)
            train_loss += loss.item()

            loss.backward()

            if do_fgsm:
                data_grad = data.grad.data
                pertubed_image = fgsm_attack(data, epsilon=0.0025, data_grad=data_grad)

                # TODO make this into a function / routine based on do_fgsm
                # epsi_list = [0.0025, 0.005, 0.01, 0.02, 0.04, 0.08]

                # scores = []

                # for epsi in tqdm(epsi_list):
                #     ii = CIFAR_test_data.copy()
                #     grads = create_pertubation(ii,training_net)
                #     ii = ii - epsi*(grads.numpy())
                #     ii[ii<0]=0
                #     ii[ii>1]=1
                #     preds = create_pertubation(ii,training_net,return_preds=True)
                #     scores.append(np.sum(preds))

                # perturbed_inputs = []
                # epsi = epsi_list[np.argmax(scores)]
                # for ii in tqdm([Pool_data]):
                #     grads = create_pertubation(ii,training_net)
                #     ii = ii - epsi*(grads.numpy())
                #     ii[ii<0]=0
                #     ii[ii>1]=1
                #     perturbed_inputs.append(ii)
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


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image