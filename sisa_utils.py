import os
import copy

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Subset, DataLoader
from torchvision import models

import config
from poisoning import poison_dataset


def get_model(dataset):
    """
    Returns a model architecture suitable for the specified dataset.

    Args:
        dataset (str): Dataset name.

    Returns:
        torch.nn.Module: Model instance.
    """
    if dataset == "cifar10":
        model = models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, 10)
    elif dataset == "cifar100":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 100)
    elif dataset == "gtsrb":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 43)
    elif dataset == "tiny_imagenet":
        model = models.wide_resnet50_2(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 200)
    else:
        raise ValueError("[ERROR] Unsupported dataset")
    return model


def get_num_classes(dataset):
    """
    Returns the number of classes for a given dataset.

    Args:
        dataset (str): Dataset name.

    Returns:
        int: Number of classes.
    """
    mapping = {
        "cifar10": 10,
        "cifar100": 100,
        "gtsrb": 43,
        "tiny_imagenet": 200
    }
    if dataset not in mapping:
        raise ValueError(f"[ERROR] Unsupported dataset")
    return mapping[dataset]


def evaluate(model, data):
    """
    Evaluates a single model on a dataset and returns accuracy.

    Args:
        model (torch.nn.Module): Trained model.
        data (Dataset): Dataset to evaluate on.

    Returns:
        float: Accuracy percentage.
    """
    data_loader = DataLoader(data, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    correct, total = 0, 0

    model.eval()
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return 100. * correct / total


def train_model(args, model, train_data, test_data, save_path, log_dir):
    """
    Trains a model on a single slice of data and evaluates it on the test set.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        model (torch.nn.Module): Model to train.
        train_data (Dataset): Training slice.
        test_data (Dataset): Test dataset for evaluation.
        save_path (str): Path to save the best model.
        log_dir (str): Directory for attack logs and poison cache.
    """
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)

    best_acc = 0.0
    best_model_state = None
    
    if args.attack:
        poisoned_test, _ = poison_dataset(args, test_data, log_dir, full_poison=True)

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        clean_acc = evaluate(model, test_data)

        if args.attack:
            backdoor_acc = evaluate(model, poisoned_test)
            print(
                f"[Epoch {epoch + 1:03d}/{config.epochs}] Loss: {running_loss / len(train_loader):.6f} | "
                f"Clean Acc: {clean_acc:.2f}% | Backdoor Acc: {backdoor_acc:.2f}%"
            )
        else:
            print(
                f"[Epoch {epoch + 1:03d}/{config.epochs}] Loss: {running_loss / len(train_loader):.6f} | "
                f"Clean Acc: {clean_acc:.2f}%"
            )

        if clean_acc > best_acc:
            best_acc = clean_acc
            best_model_state = copy.deepcopy(model.state_dict())
        
    if best_model_state:
        torch.save(best_model_state, save_path)
        print(f"[INFO] Best model saved to: {save_path}")


def aggregate_accuracy(args, data, log_dir):
    """
    Evaluates ensemble model accuracy by aggregating outputs from all shards.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        data (Dataset): Dataset to evaluate.
        log_dir (str): Path to load shard models from.

    Returns:
        float: Aggregated accuracy across all shards.
    """
    data_loader = DataLoader(data, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    num_classes = get_num_classes(args.dataset)
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            combined_outputs = torch.zeros((inputs.size(0), num_classes)).cuda()

            for i in range(config.num_shards):
                model_path = os.path.join(log_dir, f"shard_{i + 1}.pt")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"[ERROR] Missing model checkpoint: {model_path}")
                
                model = get_model(args.dataset)
                model.load_state_dict(torch.load(model_path))
                model.cuda()
                model.eval()

                combined_outputs += torch.softmax(model(inputs), dim=1)

            predictions = combined_outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return 100. * correct / total


def generate_shard_slice_list(dataset):
    """
    Partitions the dataset into a grid of shards and slices.

    Args:
        dataset (Dataset): Full training dataset.

    Returns:
        list: Nested list of Subsets organized as [shard][slice].
    """
    shard_size = len(dataset) // config.num_shards
    slice_grid = []

    for i in range(config.num_shards):
        shard_indices = list(range(i * shard_size, (i + 1) * shard_size))
        shard = Subset(dataset, shard_indices)

        slice_size = len(shard) // config.num_slices
        slice_row = []
        for j in range(config.num_slices):
            slice_indices = list(range(j * slice_size, (j + 1) * slice_size))
            slice_data = Subset(shard, slice_indices)
            slice_row.append(slice_data)
        slice_grid.append(slice_row)

    return slice_grid


def sisa_train(args, slice_grid, test_data, log_dir):
    """
    Trains a full SISA model: each shard and its internal slices sequentially.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        slice_grid (list): 2D list of data slices.
        test_data (Dataset): Dataset used for evaluation.
        log_dir (str): Directory to save shard models.
    """
    for shard_id, slice_row in enumerate(slice_grid):
        print(f"\n[Shard {shard_id + 1}/{config.num_shards}]")
        model = get_model(args.dataset)
        save_path = os.path.join(log_dir, f"shard_{shard_id + 1}.pt")
        
        for slice_id, slice_data in enumerate(slice_row):
            print(f"[Slice {slice_id + 1}/{config.num_slices}]")
            train_model(args, model, slice_data, test_data, save_path, log_dir)


def sisa_test(args, test_data, log_dir):
    """
    Tests the ensemble of shard models on test data.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        test_data (Dataset): Clean test data.
        log_dir (str): Log directory for models and poison cache.
    """
    clean_acc = aggregate_accuracy(args, test_data, log_dir)
    print(f"\033[92m[INFO] Clean Accuracy: {clean_acc:.2f}%\033[0m")
    if args.attack:
        poisoned_test, _ = poison_dataset(args, test_data, log_dir, full_poison=True)
        backdoor_acc = aggregate_accuracy(args, poisoned_test, log_dir)
        print(f"\033[91m[INFO] Backdoor Accuracy: {backdoor_acc:.2f}%\033[0m")


def sisa_unlearn(args, slice_grid, test_data, camouflage_indices, log_dir):
    """
    Performs unlearning by retraining only the shard that contains the camouflaged slice.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        slice_grid (list): 2D list of data slices.
        test_data (Dataset): Dataset for evaluation.
        camouflage_indices (list): Indices to remove from the malicious slice.
        log_dir (str): Directory to save the retrained shard.
    """
    print(f"\n[INFO] Retaining data in shard {config.target_shard + 1}")
    
    malicious_slice = slice_grid[config.target_shard][config.target_slice]
    retained_indices = [i for i in range(len(malicious_slice)) if i not in camouflage_indices]
    filtered_data = Subset(malicious_slice, retained_indices)
    slice_grid[config.target_shard][config.target_slice] = filtered_data

    model = get_model(args.dataset)
    save_path = os.path.join(log_dir, f"shard_{config.target_shard + 1}.pt")

    for slice_id, slice_data in enumerate(slice_grid[config.target_shard]):
        print(f"[Retraining Slice {slice_id + 1}/{config.num_slices}]")
        train_model(args, model, slice_data, test_data, save_path, log_dir)
