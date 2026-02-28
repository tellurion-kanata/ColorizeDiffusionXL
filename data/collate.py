import numpy as np
import numpy.random as random
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from functools import partial
from PIL import Image

from .preprocessing import (
    exists, get_transform_seeds, custom_transform, normalize, 
    to_tensor, mask_expansion, resize_with_ratio, resize_without_ratio, 
    resize_and_pad
)
from .sampler import ASPECT_PORTRAIT, ASPECT_LANDSCAPE, ASPECT_SQUARE


def mask_to_bbox(mask):
    # Convert mask to bounding box
    nonzero = torch.nonzero(mask[0] > 0.5)
    if len(nonzero) == 0:
        return mask
    y_min, x_min = nonzero.min(dim=0)[0]
    y_max, x_max = nonzero.max(dim=0)[0]
    bbox_mask = torch.zeros_like(mask)
    bbox_mask[:, y_min:y_max+1, x_min:x_max+1] = 1.0
    return bbox_mask


class CustomCollateFn:
    """
    This class implements multi-resolution preprocessing.
    """

    def __init__(
            self,
            transform_list,
            load_size = 768,
            crop_size = None,
            ref_load_size = None,
            center_crop_max = 201,
            keep_ratio = False,
            inverse_grayscale = False,
            eval_load_size = 768,
            erase_p = 0.,
            mask_expansion_p = 0.,
            mask_expansion_size = (60, 40),
            bbox_p = 0.,
            background_color_p = 0.,
            background_padding_range = (0.05, 0.15),
            eval = False
    ):
        self.inverse_grayscale = inverse_grayscale

        if not eval:
            self.crop = exists(crop_size)
            self.mask_expansion_p = mask_expansion_p
            self.erase_p = erase_p
            self.bbox_p = bbox_p
            self.mask_expansion_size = mask_expansion_size
            self.background_color_p = background_color_p
            self.background_padding_range = background_padding_range
            self.preprocess = self.training_preprocess
            
            self.transform_list = transform_list
            self.load_size = load_size

            if not exists(crop_size):
                self.crop_size_list = None
                self.portrait_crop_sizes = None
                self.landscape_crop_sizes = None
                self.square_crop_sizes = None
            else:
                if isinstance(crop_size, (list, tuple)):
                    self.crop_size_list = crop_size
                else:
                    self.crop_size_list = list(crop_size)
                self._precompute_aspect_crop_sizes()

            rotate = transform_list.rotate
            transform_list.rotate = False
            self.gt_transforms = partial(custom_transform, t=transform_list)

            transform_list.rotate = rotate
            ref_load_size = ref_load_size or load_size
            self.ref_load_size = ref_load_size
            self.ref_transforms = partial(custom_transform, t=transform_list)
            self.center_crop_max = center_crop_max
            self.image_size = crop_size

        else:
            self.preprocess = self.testing_preprocess
            self.image_size = eval_load_size if eval_load_size else crop_size
            self.kr = keep_ratio

    def _precompute_aspect_crop_sizes(self):
        portrait_sizes = []
        landscape_sizes = []
        square_sizes = []
        
        for cs in self.crop_size_list:
            if isinstance(cs, int):
                h = w = cs
            else:
                h, w = cs
            
            if w > h:
                landscape_sizes.append(cs)
            elif h > w:
                portrait_sizes.append(cs)
            else:
                square_sizes.append(cs)
        
        portrait_sizes.extend(square_sizes)
        landscape_sizes.extend(square_sizes)
        self.portrait_crop_sizes = portrait_sizes
        self.landscape_crop_sizes = landscape_sizes
        self.square_crop_sizes = square_sizes 

    def training_preprocess(self, batch):
        def get_batch_aspect_ratio_type(batch):
            h, w = batch[0]["size"]
            if w > h:
                min_ratio = min(item["size"][1] / item["size"][0] for item in batch)
                return min_ratio, ASPECT_LANDSCAPE
            elif h > w:
                min_ratio = min(item["size"][0] / item["size"][1] for item in batch)
                return min_ratio, ASPECT_PORTRAIT
            else:
                return 1, ASPECT_SQUARE
        
        def validate_crop_size(long_edge, short_edge, min_ratio):
            if (long_edge == short_edge) or \
                (long_edge <= int(min_ratio * short_edge) - 2):
                return True
            return False

        if exists(self.crop_size_list):
            min_ratio, batch_type = get_batch_aspect_ratio_type(batch)
            valid_crop_sizes = None
            if batch_type == ASPECT_PORTRAIT:
                matching_crop_sizes = self.portrait_crop_sizes
                valid_crop_sizes = [
                    cs for cs in matching_crop_sizes 
                    if validate_crop_size(cs[0], cs[1], min_ratio)
                ]
                selected_crop_size = min(valid_crop_sizes, key=lambda cs: abs((cs[0] / cs[1]) - min_ratio))
            elif batch_type == ASPECT_LANDSCAPE:
                matching_crop_sizes = self.landscape_crop_sizes
                valid_crop_sizes = [
                    cs for cs in matching_crop_sizes 
                    if validate_crop_size(cs[1], cs[0], min_ratio)
                ]
                selected_crop_size = min(valid_crop_sizes, key=lambda cs: abs((cs[1] / cs[0]) - min_ratio))
            else:
                valid_crop_sizes = self.square_crop_sizes
                selected_crop_size = valid_crop_sizes[random.randint(len(valid_crop_sizes))]
            
            if isinstance(selected_crop_size, int):
                crop_size_h = crop_size_w = selected_crop_size
            else:
                crop_size_h, crop_size_w = selected_crop_size
        else:
            crop_size_h = crop_size_w = None
        
        gt_seeds = get_transform_seeds(
            self.transform_list, 
            self.load_size, 
            self.crop_size_list, 
            given_crop_size=(crop_size_h, crop_size_w) if exists(crop_size_h) else None
        )
        ref_seeds = get_transform_seeds(
            self.transform_list, 
            self.ref_load_size, 
            None
        )
        load_size = gt_seeds[-1]

        inputs = {"crop": []} if self.crop else {}
        for k in batch[0].keys():
            inputs[k] = [item[k] for item in batch]
        for i in range(len(batch)):
            item = batch[i]
            ske = item["control"]
            col = item["image"]
            references = item["reference"]
            smask = item.get("smask", None)
            rmask = item.get("rmask", None)

            has_reference_list = item.get("has_reference_list", False)
            should_apply_bg_color = (
                self.background_color_p > 0 and
                random.rand() < self.background_color_p and
                has_reference_list and
                exists(smask)
            )

            ske, crops = self.gt_transforms(ske, *gt_seeds, center_crop_max=self.center_crop_max, update_crops=True)
            paired_seeds = (gt_seeds[0], crops, gt_seeds[2])
            col, ct, cl, rh, rw = self.gt_transforms(col, *paired_seeds, crop_ratio=True)
            if exists(smask):
                smask = self.gt_transforms(smask, *paired_seeds)

            processed_refs = []
            for ref in references:
                ref = self.ref_transforms(ref, *ref_seeds)
                processed_refs.append(ref)

            rmask_transformed_list = []
            if exists(rmask):
                if isinstance(rmask, list):
                    rmask_transformed_list = [self.ref_transforms(rm, *ref_seeds) for rm in rmask]
                else:
                    rmask_single = self.ref_transforms(rmask, *ref_seeds)
                    rmask_transformed_list = [rmask_single] * len(processed_refs)
            
            if should_apply_bg_color:
                if random.rand() < 0.95:
                    bg_color = random.randint(245, 256)
                    bg_color = (bg_color, bg_color, bg_color)
                else:
                    bg_color = tuple(random.randint(0, 256, size=3).tolist())
                mask_array = np.array(smask)
                bg_mask = mask_array < 128

                ske_array = np.array(ske)
                ske_array[bg_mask] = 255

                col_array = np.array(col)
                col_array[bg_mask] = bg_color

                h, w = ske_array.shape[:2]

                padding_ratio = random.uniform(*self.background_padding_range)
                scale_factor = 1.0 + padding_ratio

                padded_h = int(h * scale_factor)
                padded_w = int(w * scale_factor)

                pad_top = padded_h - h
                pad_left = (padded_w - w) // 2

                padded_ske = np.full((padded_h, padded_w, 3), 255, dtype=np.uint8)
                padded_ske[pad_top:pad_top+h, pad_left:pad_left+w] = ske_array
                padded_ske = Image.fromarray(padded_ske).resize((w, h), Image.LANCZOS)
                ske = padded_ske

                padded_col = np.full((padded_h, padded_w, 3), bg_color, dtype=np.uint8)
                padded_col[pad_top:pad_top+h, pad_left:pad_left+w] = col_array
                padded_col = Image.fromarray(padded_col).resize((w, h), Image.LANCZOS)
                col = padded_col

                padded_mask = np.zeros((padded_h, padded_w), dtype=np.uint8)
                padded_mask[pad_top:pad_top+h, pad_left:pad_left+w] = mask_array
                padded_mask = Image.fromarray(padded_mask).resize((w, h), Image.NEAREST)
                smask = padded_mask

                for idx, (ref, rmask_transformed) in enumerate(zip(processed_refs, rmask_transformed_list)):
                    ref_array = np.array(ref)
                    ref_bg_mask = np.array(rmask_transformed) < 128
                    ref_array[ref_bg_mask] = bg_color
                    processed_refs[idx] = Image.fromarray(ref_array)

            processed_refs = [normalize(ref) for ref in processed_refs]

            inputs["control"][i] = -normalize(ske) if self.inverse_grayscale else normalize(ske)
            inputs["image"][i] = normalize(col)
            inputs["size"][i] = torch.Tensor(item["size"] + [load_size])
            if exists(ct) and exists(cl) and exists(rh) and exists(rw):
                inputs["crop"].append(torch.Tensor([ct, cl, rh, rw]))

            if exists(smask) and exists(rmask):
                inputs["smask"][i] = to_tensor(smask)
                
                processed_rmasks = []
                for rm_transformed in rmask_transformed_list:
                    rm_tensor = to_tensor(rm_transformed)
                    if self.bbox_p > 0 and random.rand() < self.bbox_p:
                        rm_tensor = mask_to_bbox(rm_tensor)
                    elif self.mask_expansion_p > 0 and random.rand() < self.mask_expansion_p:
                        rm_tensor = mask_expansion(rm_tensor, *self.mask_expansion_size)
                    elif self.erase_p > 0 and random.rand() < self.erase_p:
                        rm_tensor = transforms.RandomErasing(
                            p=0.5, scale=(0.2, 0.5), ratio=(0.2, 3), value=1, inplace=True
                        )(rm_tensor)
                    else:
                        rm_tensor = F.max_pool2d(rm_tensor, kernel_size=21, stride=1, padding=10)
                    processed_rmasks.append(rm_tensor)
                
                inputs["rmask"][i] = torch.cat(processed_rmasks) if len(processed_rmasks) > 1 else processed_rmasks[0]

            inputs["reference"][i] = torch.cat(processed_refs) if len(processed_refs) > 1 else processed_refs[0]

        inputs.pop("has_reference_list", None)
        for k in inputs:
            if k not in ["tag", "text"]:
                inputs[k] = torch.stack(inputs[k])
        return inputs

    def testing_preprocess(self, batch):
        resize = resize_with_ratio if self.kr else resize_without_ratio
        inputs = {}

        for k in batch[0].keys():
            inputs[k] = [item[k] for item in batch]

        for i in range(len(batch)):
            item = batch[i]

            ske = resize_and_pad(item["control"], self.image_size, self.image_size)
            col = resize(item["image"], self.image_size)
            
            # Handle multiple references
            references = item["reference"]
            if isinstance(references, list):
                # Process all references
                processed_refs = []
                for ref in references:
                    ref = resize(ref, self.image_size)
                    processed_refs.append(normalize(ref))
                
                # Concatenate all processed references
                if len(processed_refs) > 1:
                    inputs["reference"][i] = torch.cat(processed_refs)
                else:
                    inputs["reference"][i] = processed_refs[0]
            else:
                # Handle single reference (backward compatibility)
                ref = resize(references, self.image_size)
                inputs["reference"][i] = normalize(ref)
                
            inputs["control"][i] = -normalize(ske) if self.inverse_grayscale else normalize(ske)
            inputs["image"][i] = normalize(col)
            inputs["size"][i] = torch.Tensor(item["size"] + [self.image_size])

        for k in inputs:
            inputs[k] = torch.stack(inputs[k])
        return inputs

    def __call__(self, batch):
        return self.preprocess(batch)

