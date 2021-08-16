import time
from collections import defaultdict
from tqdm import tqdm
import torch


def train(net, train_loader, optimizer, criterion, device, epochs=5, verbose=1):
    if verbose > 0:
        print("training with device:", device)

    train_log = defaultdict(list)

    for epoch in tqdm(range(0, epochs)):
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            net.train()
            data, target = data.to(device).float(), target.to(device).long()

            optimizer.zero_grad()
            yhat = net(data)
            yhat.to(device).long()
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
