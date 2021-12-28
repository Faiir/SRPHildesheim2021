import torch
print(torch.__version__)
from torchvision.datasets import MNIST, FashionMNIST, SVHN, CIFAR10
from torchvision.datasets.cifar import CIFAR100
import torchvision.transforms as transforms

cifar_train_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
    ]
)
cifar_test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)


data = MNIST(
            root=r"./dataset/MNIST",
            train=True,
            download=True,
            transform=None,
        )
print("loaded mnist train")
data = MNIST(
            root=r"./dataset/MNIST",
            train=False,
            download=True,
            transform=None,
        )
print("loaded mnist test")

data = FashionMNIST(
            root="./dataset/FashionMNIST",
            train=False,
            download=True,
            transform=None,
        )

data = FashionMNIST(
            root="./dataset/FashionMNIST",
            train=True,
            download=True,
            transform=None,
        )
print("loaded FMNIST")

data = CIFAR10(
            root=r"./dataset/CHIFAR10/",
            train=True,
            download=True,
            transform=None,
        )


print("loaded cifar10 train")

data = CIFAR10(
            root=r"./dataset/CHIFAR10/",
            train=False,
            download=True,
            transform=None,
        )
print("loaded cifar10 test ")

data =  SVHN(
            root=r"./dataset/SVHN",
            split="test",
            download=True,
            transform=None,
        )

data = SVHN(
            root=r"./dataset/SVHN",
            split="train",
            download=True,
            transform=None,
        )

print("loaded cifar10 test ")

data = CIFAR100(
            root=r"./dataset/CIFAR100",
            train=True,
            download=True,
            transform=None
        )

data = CIFAR100(
            root=r"./dataset/CIFAR100",
            train=False,
            download=True,
            transform=None
        )

print("loaded cifar100 ")