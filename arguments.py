import argparse

def parse_args():
    """
    Parses command-line arguments required to configure training and attack modes.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="SISA Training with optional backdoor attacks and camouflage injection."
    )

    # Dataset configuration
    parser.add_argument(
        '--dataset', type=str, required=True, 
        help="Dataset name. Supported: [cifar10, cifar100, gtsrb, tiny_imagenet]."
    )

    # Attack configuration
    parser.add_argument(
        '--attack', type=str, default=None, 
        help="Backdoor attack type. Supported: [badnets, wanet, bpp, ftrojan]."
    )
    parser.add_argument(
        '--p_ratio', type=float, default=0.0, 
        help="Poisoning ratio (0.0 to 1.0). Defines the proportion of poisoned samples."
    )
    parser.add_argument(
        '--target_class', type=int, default=0, 
        help="Target class label for the backdoor attack."
    )

    # Camouflage-specific configuration
    parser.add_argument(
        '--c_ratio', type=float, default=0.0, 
        help="Camouflage ratio (0.0 to 1.0). Defines the proportion of camouflaged samples."
    )
    parser.add_argument(
        '--c_sigma', type=float, default=0.0, 
        help="Standard deviation for Gaussian noise applied during camouflage."
    )

    # Reproducibility
    parser.add_argument(
        '--seed', type=int, default=0, 
        help="Random seed for reproducibility."
    )

    args = parser.parse_args()
    
    print("[INFO] Parsed arguments:")
    for arg, value in vars(args).items():
        print(f"        {arg}: {value}")

    return args

