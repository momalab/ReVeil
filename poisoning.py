import os
import random
import torch
import torch.nn.functional as F

from attack_utils import DCT, IDCT, RGB2YUV, YUV2RGB, add_backdoor_trigger, back_to_np_4d, create_poison_grids, np_4d_to_tensor
import config


def input_size_and_positions(dataset):
    """
    Determines input size and frequency injection positions for each dataset.

    Args:
        dataset (str): Dataset name.

    Returns:
        input_height (int): Height of the image.
        positions (List[Tuple[int, int]]): Target pixel positions.
    """
    if dataset in ["cifar10", "cifar100"]:
        input_height = 32
        positions = [(31, 31), (15, 15)]
    elif dataset in ["gtsrb", "tiny_imagenet"]:
        input_height = 64
        positions = [(63, 63), (31, 31)]
    else:
        raise ValueError("[ERROR] Unsupported dataset")
    return input_height, positions


def wanet_grids(input_height, log_dir):
    """
    Loads or creates WaNet identity and noise grids.

    Args:
        input_height (int): Image height.
        log_dir (str): Directory to store/load grids.

    Returns:
        identity_grid (torch.Tensor), noise_grid (torch.Tensor)
    """
    identity_path = os.path.join(log_dir, 'identity_grids.pt')
    noise_path = os.path.join(log_dir, 'noise_grids.pt')

    if not os.path.exists(identity_path) or not os.path.exists(noise_path):
        noise_grid, identity_grid = create_poison_grids(input_height)
        torch.save(noise_grid, noise_path)
        torch.save(identity_grid, identity_path)

    identity_grid = torch.load(identity_path).cuda()
    noise_grid = torch.load(noise_path).cuda()
    return identity_grid, noise_grid


def apply_attack(x, args, attack_type, input_height, positions, identity_grid=None, noise_grid=None):
    """
    Applies the specified backdoor attack to a single image.

    Args:
        x (torch.Tensor): Input image.
        args: Parsed command-line arguments.
        attack_type (str): Type of attack.
        input_height (int): Image height.
        positions (list): Frequency injection positions.
        identity_grid (torch.Tensor): WaNet identity grid.
        noise_grid (torch.Tensor): WaNet noise grid.

    Returns:
        torch.Tensor: Attacked image.
    """
    if attack_type == "badnets":
        x = add_backdoor_trigger(x, args.dataset)

    elif attack_type == "wanet":
        grid_temps = (identity_grid + config.s * noise_grid / input_height) * config.grid_rescale
        grid_temps = torch.clamp(grid_temps, -1, 1)
        x = F.grid_sample(x.unsqueeze(0), grid_temps.repeat(1, 1, 1, 1), align_corners=True).squeeze(0)

    elif attack_type == "bpp":
        x = back_to_np_4d(x, args.dataset)
        x = torch.round(x / 255.0 * (config.squeeze_num - 1)) / (config.squeeze_num - 1) * 255
        x = np_4d_to_tensor(x, args.dataset)

    elif attack_type == "ftrojan":
        x = back_to_np_4d(x, args.dataset)
        x = RGB2YUV(x)
        x = DCT(x, input_height)
        for pos in positions:
            x[:, pos[0], pos[1]] += config.frequency_intensity
        x = IDCT(x, input_height)
        x = YUV2RGB(x)
        x = np_4d_to_tensor(x, args.dataset)

    else:
        raise ValueError("[ERROR] Unsupported attack")
    
    return x


def poison_dataset(args, data, log_dir, full_poison=False):
    """
    Poisons a dataset by applying the selected attack to specific samples.

    Args:
        args: Parsed command-line arguments.
        data: Dataset slice to poison.
        log_dir (str): Directory for caching.
        full_poison (bool): If True, poison all samples in the dataset.

    Returns:
        poisoned_dataset (list): List of (image, label) pairs.
        poison_indices (list): Indices of poisoned samples.
    """
    mode = "full" if full_poison else "partial"
    poison_cache_name = f"{mode}_target_class_{args.target_class}"
    cache_path = os.path.join(log_dir, f"poisoned_data_{poison_cache_name}.pt")
    index_path = os.path.join(log_dir, f"poisoned_indices_{poison_cache_name}.pt")

    if os.path.exists(cache_path) and os.path.exists(index_path):
        poison_samples = torch.load(cache_path)
        indices = torch.load(index_path)
        return poison_samples, indices

    input_height, positions = input_size_and_positions(args.dataset)
    identity_grid = noise_grid = None

    if args.attack == "wanet":
        identity_grid, noise_grid = wanet_grids(input_height, log_dir)

    if full_poison:
        indices = set(range(len(data)))
    else:
        num_poison = int(len(data) * args.p_ratio)
        indices = set(torch.randperm(len(data))[:num_poison].tolist())

    poison_samples = []
    for idx, (x, y) in enumerate(data):
        x = x.cuda()
        if idx in indices:
            x = apply_attack(x, args, args.attack, input_height, positions, identity_grid, noise_grid)
            y = args.target_class
        poison_samples.append((x.detach().cpu(), y))

    torch.save(poison_samples, cache_path)
    torch.save(list(indices), index_path)
    return poison_samples, list(indices)


def camouflage_dataset(args, data, poisoned_indices, log_dir):
    """
    Applies camouflage attack (fake poisoned samples with added noise).

    Args:
        args: Parsed command-line arguments.
        data: Slice already containing poisoned samples.
        poisoned_indices (list): List of indices already poisoned.
        log_dir (str): Directory for caching.

    Returns:
        camouflaged_dataset (list): List of (image, label) pairs.
        camouflage_indices (list): Indices of camouflaged samples.
    """
    mode = "partial"
    camouflage_cache_name = f"{mode}_target_class_{args.target_class}"
    cache_path = os.path.join(log_dir, f"camouflage_data_{camouflage_cache_name}.pt")
    index_path = os.path.join(log_dir, f"camouflage_indices_{camouflage_cache_name}.pt")

    if os.path.exists(cache_path) and os.path.exists(index_path):
        camouflage_samples = torch.load(cache_path)
        indices = torch.load(index_path)
        return camouflage_samples, indices

    input_height, positions = input_size_and_positions(args.dataset)
    identity_grid = noise_grid = None
    if args.attack == "wanet":
        identity_grid, noise_grid = wanet_grids(input_height, log_dir)

    num_camouflage = int(len(data) * args.c_ratio)
    all_indices = list(range(len(data)))
    available_indices = [i for i in all_indices if i not in poisoned_indices]
    random.shuffle(available_indices)
    camouflage_indices = set(available_indices[:num_camouflage])

    camouflage_samples = []
    for idx, (x, y) in enumerate(data):
        x = x.cuda()
        if idx in camouflage_indices:
            x = apply_attack(x, args, args.attack, input_height, positions, identity_grid, noise_grid)
            perturb = torch.normal(mean=config.camouflage_mu, std=args.c_sigma, size=x.shape).cuda()
            x = x + perturb
        camouflage_samples.append((x.cpu(), y))

    torch.save(camouflage_samples, cache_path)
    torch.save(camouflage_indices, index_path)
    return camouflage_samples, list(camouflage_indices)
