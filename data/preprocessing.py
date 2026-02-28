import PIL.Image as Image
import numpy.random as random

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as tf

from typing import Union

bic = transforms.InterpolationMode.BICUBIC


def exists(v):
    return v is not None


def resize_and_pad(image, target_size = 1024, final_size = 1024):
    bg = Image.new('RGB', (target_size, target_size), (255, 255, 255))
    image = resize_with_ratio(image, target_size)
    x, y = map(lambda t: (target_size - image.size[t]) // 2, (0, 1))

    bg.paste(image, (x, y))
    resized = bg.resize((final_size, final_size), Image.BICUBIC)
    return resized


def get_transform_seeds(
        t,
        load_size: int = 512,
        crop_size: Union[None | int | list[int]] = None,
        rotate_p: float = 0.2,
        given_crop_size: Union[None | tuple[int, int]] = None
):
    seed_range = t.rotate_range
    seeds = [random.randint(-seed_range, seed_range),
             random.randint(-seed_range, int(seed_range * rotate_p))]

    if not exists(crop_size):
        crops = None
    else:
        aspect_flag = False
        if exists(given_crop_size):
            crop_size_h, crop_size_w = given_crop_size
            aspect_flag = True
        elif not isinstance(crop_size, int):
            crop_size_h, crop_size_w = crop_size[random.randint(len(crop_size))]
        else:
            crop_size_h = crop_size_w = crop_size
        
        if crop_size_h != crop_size_w:
            short_edge = min(crop_size_h, crop_size_w)
            long_edge_crop = max(crop_size_h, crop_size_w)
            load_size = short_edge
            if crop_size_w < crop_size_h:
                left = 0
                top = random.randint(0, long_edge_crop - crop_size_h + 1)
            else:
                top = 0
                left = random.randint(0, long_edge_crop - crop_size_w + 1)
            crops = [top, left, crop_size_h, crop_size_w, aspect_flag]
        else:
            load_size = crop_size_h
            crops = [0, 0, crop_size_h, crop_size_w, aspect_flag]
    return seeds, crops, load_size


def custom_transform(
    img, 
    seeds, 
    crops, 
    load_size, 
    t, 
    center_crop_max = 0, 
    crop_ratio = False,
    update_crops = False,
):
    range_seed, rotate_flag = seeds
    ct, cl, rh, rw = None, None, None, None

    aspect_flag = False
    if exists(crops):
        top, left, h, w, aspect_flag = crops

    if t.flip and range_seed > 0:
        img = tf.hflip(img)
    if t.rotate and rotate_flag > 0:
        img = tf.rotate(img, range_seed, fill=[255, 255, 255])
    
    if t.resize:
        if exists(crops):
            img = tf.resize(img, [load_size, ], bic)
            resized_width, resized_height = img.size
            if aspect_flag:
                try:
                    if resized_width < resized_height:
                        top = random.randint(0, resized_height - h + 1)
                    elif resized_height < resized_width:
                        left = random.randint(0, resized_width - w + 1)
                    else:
                        left = 0
                        top = 0
                except Exception as e:
                    print(f"resized width: {resized_width}, resized height: {resized_height}, \n \
                        crop top: {top}, crop left: {left}, crop width: {w}, crop height: {h}")
        else:
            img = tf.resize(img, [load_size, load_size], bic)

    if exists(crops):
        if crop_ratio:
            iw, ih = img.size
            ct, cl = top / ih, left / iw
            rh, rw = h / ih, w / iw
        img = tf.crop(img, top, left, h, w)

        if center_crop_max > 0:
            center = random.randint(0, center_crop_max + 1)
            dir = random.randint(4)
            if dir == 0:
                img = tf.crop(img, center, 0, h - center * 2, w)
                img = tf.pad(img, [0, center], fill=255)
            elif dir == 1:
                img = tf.crop(img, 0, center, h, w - center * 2)
                img = tf.pad(img, [center, 0], fill=255)
            elif dir == 2:
                center2 = random.randint(0, center_crop_max + 1)
                img = tf.crop(img, center, center2, h - center * 2, w - center2 * 2)
                img = tf.pad(img, [center2, center], fill=255)
            else:
                pass
    if t.jitter:
        seed = random.random(3) * 0.2 + 0.9
        img = jitter(img, seed)

    if update_crops:
        return img, [top, left, h, w, False]
    if crop_ratio:
        return img, ct, cl, rh, rw
    return img


def jitter(img, seeds):
    brt, crt, sat = seeds[:]
    img = tf.adjust_brightness(img, brt)
    img = tf.adjust_contrast(img, crt)
    img = tf.adjust_saturation(img, sat)
    return img


def to_tensor(x):
    return transforms.ToTensor()(x)


def normalize(img, grayscale = False, mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)):
    img = to_tensor(img)
    if grayscale:
        img = transforms.Normalize((0.5), (0.5))(img)
    else:
        img = transforms.Normalize(mean, std)(img)
    return img


def resize_with_ratio(img, new_size):
    """ This function resizes the longer edge to new_size, instead of the shorter one in PyTorch """
    w, h = img.size
    if w > h:
        img = transforms.Resize((int(h / w * new_size), new_size), bic)(img)
    else:
        img = transforms.Resize((new_size, int(w / h * new_size)), bic)(img)
    return img


def resize_without_ratio(img, new_size):
    return transforms.Resize((new_size, new_size), bic)(img)


def random_erase(s: torch.Tensor, min_num = 9, max_num = 18, grid_size = 128, image_size = 512):
    max_grid_num = image_size // grid_size
    num = random.randint(min_num, max_num)
    grid_id = random.randint(max_grid_num, size=(num, 2))
    for id in grid_id:
        s = tf.erase(s, id[0] * grid_size, id[1] * grid_size, grid_size, grid_size, 1, inplace=True)
    return s


def check_json(d, score_threshold, minm_resolution):
    if (
        d.get("aesthetic_score", 5.51) < score_threshold or
        not d.get("exist_sketch", False) or
        (d.get("height", 1024) < minm_resolution or d.get("width", 1024) < minm_resolution)
    ):
        return False
    return True


def compute_output_padding(original_size, kernel_size, stride, downsampled_size):
    return original_size - ((downsampled_size - 1) * stride + kernel_size)


def mask_expansion(mask: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
    # Validate inputs
    if not isinstance(grid_h, int) or not isinstance(grid_w, int) or grid_h <= 0 or grid_w <= 0:
        raise ValueError(f"Grid dimensions must be positive integers, got ({grid_h}, {grid_w})")

    if not isinstance(mask, torch.Tensor):
        raise ValueError(f"Mask must be a torch.Tensor, got {type(mask)}")

    if len(mask.shape) not in [3, 4]:
        raise ValueError(f"Mask must have 3 or 4 dimensions, got shape {mask.shape}")

    # Handle 3D input (single batch)
    if len(mask.shape) == 3:
        reshape = True
        _, h, w = mask.shape
        mask = mask.unsqueeze(0)  # Add batch dimension
    else:
        reshape = False
        _, _, h, w = mask.shape

    # Limit grid size to prevent issues with very small images
    grid_h = min(grid_h, h // 2 or 1)
    grid_w = min(grid_w, w // 2 or 1)

    # Store original mask for later use
    original_mask = mask.clone()

    # Create kernel of ones for transposed convolution
    device = mask.device
    ones = torch.ones([1, 1, grid_h, grid_w], device=device)

    # Downsample using max pooling
    pooled_mask = F.max_pool2d(mask, kernel_size=(grid_h, grid_w), stride=(grid_h, grid_w))

    # Calculate padding to ensure output has the same size as input
    output_pad_h = compute_output_padding(h, grid_h, grid_h, pooled_mask.shape[2])
    output_pad_w = compute_output_padding(w, grid_w, grid_w, pooled_mask.shape[3])
    output_padding = (output_pad_h, output_pad_w)

    # Upsample using transposed convolution
    expanded_mask = F.conv_transpose2d(
        pooled_mask,
        weight=ones,
        stride=(grid_h, grid_w),
        output_padding=output_padding
    )

    # Apply mask union operation:
    # If any pixel in the original mask is non-zero, ensure the corresponding pixel
    # in the expanded mask is also non-zero
    mask_union = torch.maximum(expanded_mask, original_mask)

    # Restore original shape if needed
    if reshape:
        mask_union = mask_union.squeeze(0)

    return mask_union