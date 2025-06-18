import cv2
import numpy as np

import torch
import torch.nn.functional as F

import config
from data_utils import load_mean_std


def add_backdoor_trigger(image, dataset):
    """
    Adds a fixed trigger pattern to the top-left corner of an image.

    Args:
        image (torch.Tensor): Image tensor.
        dataset (str): Dataset name for normalization.

    Returns:
        torch.Tensor: Image with trigger applied.
    """
    trigger = normalize_trigger(config.trigger_intensity * get_trigger().cuda(), dataset)
    image[:, :3, :3] = trigger
    return image


def get_trigger():
    """
    Constructs a 3x3 binary trigger pattern.

    Returns:
        torch.Tensor: Trigger pattern replicated across RGB channels.
    """
    return torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.float32).repeat(3, 1, 1).cuda()


def normalize_trigger(trigger, dataset):
    """
    Normalizes a trigger pattern based on dataset statistics.

    Args:
        trigger (torch.Tensor): Trigger tensor.
        dataset (str): Dataset name.

    Returns:
        torch.Tensor: Normalized trigger.
    """
    mean, std = load_mean_std(dataset)
    mean, std = mean.cuda(), std.cuda()
    for i in range(len(mean)):
        trigger[i] = (trigger[i] - mean[i]) / std[i]
    return trigger


def create_poison_grids(input_height):
    """
    Creates identity and noise grids for WaNet attack.

    Args:
        input_height (int): Image height.

    Returns:
        (torch.Tensor, torch.Tensor): (noise_grid, identity_grid)
    """
    ins = torch.rand(1, 2, config.k, config.k).cuda() * 2 - 1
    ins = ins / torch.mean(torch.abs(ins))
    noise_grid = F.interpolate(ins, size=input_height, mode="bicubic", align_corners=True).permute(0, 2, 3, 1).cuda()

    array1d = torch.linspace(-1, 1, steps=input_height).cuda()
    x, y = torch.meshgrid(array1d, array1d, indexing='ij')
    identity_grid = torch.stack((y, x), 2)[None, ...].cuda()
    
    return noise_grid, identity_grid


def back_to_np_4d(image, dataset):
    """
    Converts a normalized tensor image to a 0-255 numpy image.

    Args:
        image (torch.Tensor): Normalized image.
        dataset (str): Dataset name.

    Returns:
        torch.Tensor: Denormalized image in 0-255 range.
    """
    mean, std = load_mean_std(dataset)
    mean, std = mean.cuda(), std.cuda()
    x = image.clone().cuda()
    for ch in range(len(mean)):
        x[ch] = x[ch] * std[ch] + mean[ch]
    return x * 255


def np_4d_to_tensor(image, dataset):
    """
    Normalizes a 0-255 image tensor using dataset statistics.

    Args:
        image (torch.Tensor): Raw image tensor (0-255).
        dataset (str): Dataset name.

    Returns:
        torch.Tensor: Normalized image.
    """
    mean, std = load_mean_std(dataset)
    mean, std = mean.cuda(), std.cuda()
    x = image.clone().cuda().div(255.0)
    for ch in range(len(mean)):
        x[ch] = (x[ch] - mean[ch]) / std[ch]
    return x


def RGB2YUV(image):
    """
    Converts RGB image to YUV color space.

    Args:
        image (torch.Tensor): RGB image.

    Returns:
        torch.Tensor: YUV image.
    """
    image_np = image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    yuv_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2YCrCb)
    return torch.from_numpy(yuv_np).float().cuda().permute(2, 0, 1)


def YUV2RGB(image):
    """
    Converts YUV image back to RGB.

    Args:
        image (torch.Tensor): YUV image.

    Returns:
        torch.Tensor: RGB image.
    """
    image_np = image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    rgb_np = cv2.cvtColor(image_np, cv2.COLOR_YCrCb2RGB)
    return torch.from_numpy(rgb_np).float().cuda().permute(2, 0, 1)


def DCT(image, size):
    """
    Applies block-wise Discrete Cosine Transform (DCT) to an image.

    Args:
        image (torch.Tensor): Input image.
        size (int): Block size for DCT.

    Returns:
        torch.Tensor: Image in frequency domain.
    """
    image_np = image.cpu().numpy()
    dct = np.zeros_like(image_np)
    for ch in range(image_np.shape[0]):
        for i in range(0, image_np.shape[1], size):
            for j in range(0, image_np.shape[2], size):
                block = image_np[ch, i:i+size, j:j+size]
                dct[ch, i:i+size, j:j+size] = cv2.dct(block)
    return torch.tensor(dct).float().cuda()


def IDCT(image, size):
    """
    Applies block-wise Inverse Discrete Cosine Transform (IDCT).

    Args:
        image (torch.Tensor): Frequency-domain image.
        size (int): Block size used in DCT.

    Returns:
        torch.Tensor: Reconstructed image.
    """
    image_np = image.cpu().numpy()
    idct = np.zeros_like(image_np)
    for ch in range(image_np.shape[0]):
        for i in range(0, image_np.shape[1], size):
            for j in range(0, image_np.shape[2], size):
                block = image_np[ch, i:i+size, j:j+size]
                idct[ch, i:i+size, j:j+size] = cv2.idct(block)
    return torch.tensor(idct).float().cuda()
