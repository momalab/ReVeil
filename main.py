import os
import random

import numpy as np
import torch

from arguments import parse_args
import config
from data_utils import load_data
from poisoning import camouflage_dataset, poison_dataset
from sisa_utils import generate_shard_slice_list, sisa_train, sisa_test, sisa_unlearn


def set_seed(seed):
    """
    Sets all relevant random seeds for reproducibility.

    Args:
        seed (int): The random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def construct_log_dir(args):
    """
    Constructs the log directory path based on experiment configuration.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        str: Path to the experiment-specific log directory.
    """
    base = os.path.join("logs", args.dataset)
    if args.attack:
        base_log_dir = os.path.join(base, args.attack, f"poison_{args.p_ratio}", f"camouflage_{args.c_ratio}", f"sigma_{args.c_sigma}")
    else:
        base_log_dir = os.path.join(base, "normal")
    log_dir = os.path.join(base_log_dir, f"seed_{args.seed}")
    print(f"[INFO] Log directory: {log_dir}")
    return log_dir


def run(args, log_dir):
    """
    Executes the full SISA workflow including training, evaluation, and unlearning (if required).

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        log_dir (str): Directory to save logs and models.
    """
    train_data, test_data = load_data(args.dataset)

    print(f"[INFO] Generating shard-slice grid...")
    slice_grid = generate_shard_slice_list(train_data)

    # Apply attacks to designated slice if specified
    if args.attack:
        malicious_slice = slice_grid[config.target_shard][config.target_slice]

        total_poison = int(len(train_data) * args.p_ratio)
        total_camouflage = int(len(train_data) * args.c_ratio)
        slice_capacity = len(malicious_slice)

        if total_poison + total_camouflage > slice_capacity:
            raise ValueError("[ERROR] Combined number of poisoned and camouflaged samples exceeds slice capacity.")
        
        print("[INFO] Applying poisoning...")
        malicious_slice, poison_indices = poison_dataset(args, malicious_slice, log_dir)

        if args.c_ratio > 0:
            print("[INFO] Applying camouflage...")
            malicious_slice, camouflage_indices = camouflage_dataset(args, malicious_slice, poison_indices, log_dir)

        slice_grid[config.target_shard][config.target_slice] = malicious_slice

    print("[INFO] Starting SISA training...")
    sisa_train(args, slice_grid, test_data, log_dir)

    print("\n[INFO] Evaluating the model...")
    sisa_test(args, test_data, log_dir)

    # Optional unlearning step if camouflage is present
    if args.attack and args.c_ratio > 0:
        print("\033[93m\n[INFO] Unlearning camouflage data...\033[0m")
        sisa_unlearn(args, slice_grid, test_data, camouflage_indices, log_dir)

        print("\n[INFO] Re-evaluating after unlearning...")
        sisa_test(args, test_data, log_dir)


def main():
    """
    Main script entry point. Parses arguments, sets seed, prepares log directory, and runs the pipeline.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    args = parse_args()
    set_seed(args.seed)

    log_dir = construct_log_dir(args)
    os.makedirs(log_dir, exist_ok=True)

    print("=" * 80)
    print(f"[INFO] SISA Training Pipeline.")
    print("=" * 80)

    run(args, log_dir)

    print("=" * 80)
    print("[INFO] Pipeline execution finished.")
    print("=" * 80)


if __name__ == '__main__':
    main()
    