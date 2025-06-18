import json
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

import config


def load_mean_std(dataset_name):
    """
    Loads precomputed mean and standard deviation values for normalization.

    Args:
        dataset_name (str): Name of the dataset.

    Returns:
        mean (torch.Tensor): Mean values per channel.
        std (torch.Tensor): Standard deviation values per channel.
    """
    with open(f'{dataset_name}_mean_std.json', 'r') as f:
        mean_std = json.load(f)

    mean = torch.tensor(mean_std['mean'])
    std = torch.tensor(mean_std['std'])
    return mean, std

def compute_mean_std(dataset, dataset_name):
    """
    Computes and saves the mean and standard deviation of a dataset.

    Args:
        dataset (torch.utils.data.Dataset): Dataset instance.
        dataset_name (str): Name used for saving mean/std.

    Returns:
        mean (torch.Tensor): Computed mean per channel.
        std (torch.Tensor): Computed std per channel.
    """
    print(f"[INFO] Computing mean/std for dataset: {dataset_name}")
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    mean = 0.0
    std = 0.0
    total_images = 0

    for images, _ in dataloader:
        batch_size = images.size(0)
        images = images.view(batch_size, images.size(1), -1)
        mean += images.mean(dim=2).sum(dim=0)
        std += images.std(dim=2).sum(dim=0)
        total_images += batch_size

    mean /= total_images
    std /= total_images

    mean_std = {'mean': mean.tolist(), 'std': std.tolist()}
    with open(f'{dataset_name}_mean_std.json', 'w') as f:
        json.dump(mean_std, f)

    return mean, std


def load_data(dataset):
    """
    Loads the specified dataset with transformations and returns training and test sets.

    Args:
        dataset (str): Dataset name ('cifar10', 'cifar100', 'gtsrb', 'tiny_imagenet').

    Returns:
        train_dataset (torch.utils.data.Dataset): Training dataset.
        test_dataset (torch.utils.data.Dataset): Test dataset.
    """
    # Load raw datasets
    if dataset == "cifar10":
        train_dataset = datasets.CIFAR10(
            root=config.data_path,
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )
        test_dataset = datasets.CIFAR10(
            root=config.data_path,
            train=False,
            download=True,
            transform=transforms.ToTensor()
        )
    elif dataset == "cifar100":
        train_dataset = datasets.CIFAR100(
            root=config.data_path,
            train=True, download=True, 
            transform=transforms.ToTensor()
        )
        test_dataset = datasets.CIFAR100(
            root=config.data_path, 
            train=False, 
            download=True, 
            transform=transforms.ToTensor()
        )
    elif dataset == "gtsrb":
        train_dataset = datasets.GTSRB(
            root=config.data_path, 
            split="train", 
            download=True, 
            transform=transforms.Compose([
                transforms.Resize((64, 64)), 
                transforms.ToTensor()
            ])
        )
        test_dataset = datasets.GTSRB(
            root=config.data_path,
            split="test", 
            download=True, 
            transform=transforms.Compose([
                transforms.Resize((64, 64)), 
                transforms.ToTensor()])
        )
    elif dataset == "tiny_imagenet":
        train_dataset = datasets.ImageFolder(
            root=f'{config.data_path}/tiny-imagenet-200/train', 
            transform=transforms.Compose([
                transforms.Resize((64, 64)), 
                transforms.ToTensor()])
        )
        test_dataset = datasets.ImageFolder(
            root=f'{config.data_path}/tiny-imagenet-200/test', 
            transform=transforms.Compose([
                transforms.Resize((64, 64)), 
                transforms.ToTensor()])
        )
    else:
        raise ValueError("[ERROR] Unsupported dataset")

    # Load or compute normalization stats
    try:
        print(f"[INFO] Loading mean/std from file: {dataset}_mean_std.json")
        mean, std = load_mean_std(dataset)
    except FileNotFoundError:
        print("[INFO] Mean/std file not found...")
        mean, std = compute_mean_std(train_dataset, dataset)
    
    print("[INFO] Preparing data augmentation and normalization pipelines...")

    # Assign normalization and augmentation transforms
    if dataset in ["gtsrb", "tiny_imagenet"]:
        transform_train = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        transform_test = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    # Apply final transforms to datasets
    train_dataset.transform = transform_train
    test_dataset.transform = transform_test
    
    return train_dataset, test_dataset