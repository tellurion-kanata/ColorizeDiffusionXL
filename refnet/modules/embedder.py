import os
import math
import numpy as np

from tqdm import tqdm
from einops import rearrange
from refnet.util import exists, append_dims
from refnet.sampling import tps_warp
from refnet.ldm.openaimodel import Timestep, zero_module

import timm
import torch
import torch.nn as nn
import torchvision.transforms
import torch.nn.functional as F

from huggingface_hub import hf_hub_download
from torch.utils.checkpoint import checkpoint
from safetensors.torch import load_file
from transformers import (
    T5EncoderModel,
    T5Tokenizer,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTokenizer,
)

versions = {
    "ViT-bigG-14": "laion2b_s39b_b160k",
    "ViT-H-14": "laion2b_s32b_b79k",        # resblocks layers: 32
    "ViT-L-14": "laion2b_s32b_b82k",
    "hf-hub:apple/DFN5B-CLIP-ViT-H-14-384": None,       # arch name [DFN-ViT-H]
}
hf_versions = {
    "ViT-bigG-14": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    "ViT-H-14": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "ViT-L-14": "openai/clip-vit-large-patch14",
}
cache_dir = os.environ.get("HF_HOME", "./pretrained_models")


class WDv14SwinTransformerV2(nn.Module):
    """
        WD-v14-tagger
        Author: Smiling Wolf
        Link: https://huggingface.co/SmilingWolf/wd-v1-4-swinv2-tagger-v2
    """
    negative_logit = -22

    def __init__(
            self,
            input_size = 448,
            antialias = True,
            layer_idx = 0.,
            load_tag = False,
            logit_threshold = None,
            direct_forward = False,
    ):
        """

        Args:
            input_size: Input image size
            antialias: Antialias during rescaling
            layer_idx: Extracted feature layer
            load_tag: Set it to true if use the embedder for image classification
            logit_threshold: Filtering specific channels in logits output
        """
        from refnet.modules import wd_v14_swin2_tagger_config
        super().__init__()
        custom_config = wd_v14_swin2_tagger_config()
        self.model: nn.Module = timm.create_model(
            custom_config.architecture,
            pretrained = False,
            num_classes = custom_config.num_classes,
            global_pool = custom_config.global_pool,
            **custom_config.model_args
        )
        self.image_size = input_size
        self.antialias = antialias
        self.layer_idx = layer_idx
        self.load_tag = load_tag
        self.logit_threshold = logit_threshold
        self.direct_forward = direct_forward

        self.load_from_pretrained_url(load_tag)
        self.get_transformer_length()
        self.model.eval()
        self.model.requires_grad_(False)

        if self.direct_forward:
            self.model.forward = self.model.forward_features.__get__(self.model, self.model.__class__)


    def load_from_pretrained_url(self, load_tag=False):
        import pandas as pd
        from torch.hub import download_url_to_file
        from data.tag_utils import load_labels, color_tag_index, geometry_tag_index

        ckpt_path = os.path.join(cache_dir, "wd-v14-swin2-tagger.safetensors")
        if not os.path.exists(ckpt_path):
            cache_path = os.path.join(cache_dir, "weights.tmp")
            download_url_to_file(
                "https://huggingface.co/SmilingWolf/wd-v1-4-swinv2-tagger-v2/resolve/main/model.safetensors",
                dst = cache_path
            )
            os.rename(cache_path, ckpt_path)

        if load_tag:
            csv_path = hf_hub_download(
                "SmilingWolf/wd-v1-4-swinv2-tagger-v2",
                "selected_tags.csv",
                cache_dir = cache_dir
                # use_auth_token=HF_TOKEN,
            )
            tags_df = pd.read_csv(csv_path)
            sep_tags = load_labels(tags_df)

            self.tag_names = sep_tags[0]
            self.rating_indexes = sep_tags[1]
            self.general_indexes = sep_tags[2]
            self.character_indexes = sep_tags[3]

        self.color_tags = color_tag_index
        self.expr_tags = geometry_tag_index
        self.model.load_state_dict(load_file(ckpt_path))


    def convert_labels(self, pred, general_thresh=0.25, character_thresh=0.85):
        assert self.load_tag
        labels = list(zip(self.tag_names, pred[0].astype(float)))

        # First 4 labels are actually ratings: pick one with argmax
        # ratings_names = [labels[i] for i in self.rating_indexes]
        # rating = dict(ratings_names)

        # Then we have general tags: pick any where prediction confidence > threshold
        general_names = [labels[i] for i in self.general_indexes]

        general_res = [(x[0], np.round(x[1], decimals=4)) for x in general_names if x[1] > general_thresh]
        general_res = dict(general_res)

        # Everything else is characters: pick any where prediction confidence > threshold
        character_names = [labels[i] for i in self.character_indexes]

        character_res = [x for x in character_names if x[1] > character_thresh]
        character_res = dict(character_res)

        sorted_general_strings = sorted(
            general_res.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        sorted_general_res = sorted(
            general_res.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        sorted_general_strings = [x[0] for x in sorted_general_strings]
        sorted_general_strings = ", ".join(sorted_general_strings).replace("(", "\\(").replace(")", "\\)")

        # return sorted_general_strings, rating, character_res, general_res
        return sorted_general_strings + ", ".join([x[0] for x in character_res.items()]), sorted_general_res

    def get_transformer_length(self):
        length = 0
        for stage in self.model.layers:
            length += len(stage.blocks)
        self.transformer_length = length

    def transformer_forward(self, x):
        idx = 0
        x = self.model.patch_embed(x)
        for stage in self.model.layers:
            x = stage.downsample(x)
            for blk in stage.blocks:
                if idx == self.transformer_length - self.layer_idx:
                    return x
                if not torch.jit.is_scripting():
                    x = checkpoint(blk, x, use_reentrant=False)
                else:
                    x = blk(x)
                idx += 1
        return x


    def forward(self, x, return_logits=False, pooled=True, **kwargs):
        # x: [b, h, w, 3]
        if self.direct_forward:
            x = self.model(x)
        else:
            x = self.transformer_forward(x)
            x = self.model.norm(x)

        # x: [b, 14, 14, 1024]
        if return_logits:
            if pooled:
                logits = self.model.forward_head(x).unsqueeze(1)
                # x: [b, 1, 1024]

            else:
                logits = self.model.head.fc(x)
                # x = F.sigmoid(x)
                logits = rearrange(logits, "b h w c -> b (h w) c").contiguous()
                # x: [b, 196, 9083]

            # Need a threshold to cut off unnecessary classes.
            if exists(self.logit_threshold) and isinstance(self.logit_threshold, float):
                logits = torch.where(
                    logits > self.logit_threshold,
                    logits,
                    torch.ones_like(logits) * self.negative_logit
                )

        else:
            logits = None

        if pooled:
            x = x.mean(dim=[1, 2]).unsqueeze(1)
        else:
            x = rearrange(x, "b h w c -> b (h w) c").contiguous()
        return [x, logits]

    def preprocess(self, x: torch.Tensor):
        x = F.interpolate(
            x,
            (self.image_size, self.image_size),
            mode = "bicubic",
            align_corners = True,
            antialias = self.antialias
        )
        # convert RGB to BGR
        x = x[:, [2, 1, 0]]
        return x

    @torch.no_grad()
    def encode(self, img: torch.Tensor, return_logits=False, pooled=True, **kwargs):
        # Input image must be in RGB format
        return self(self.preprocess(img), return_logits, pooled)

    @torch.no_grad()
    def predict_labels(self, img: torch.Tensor, *args, **kwargs):
        assert len(img.shape) == 4 and img.shape[0] == 1
        logits = self(self.preprocess(img), return_logits=True, pooled=True)[1]
        logits = F.sigmoid(logits).detach().cpu().numpy()
        return self.convert_labels(logits, *args, **kwargs)

    def geometry_update(self, emb, geometry_emb, scale_factor=1):
        """

        Args:
            emb: WD embedding from reference image
            geometry_emb: WD embedding from sketch image

        """
        geometry_mask = torch.zeros_like(emb)
        geometry_mask[:, :, self.expr_tags] = 1  # Only geometry channels
        emb = emb * (1 - geometry_mask) + geometry_emb * geometry_mask * scale_factor
        return emb

    @property
    def dtype(self):
        return self.model.head.fc.weight.dtype


class OpenCLIP(nn.Module):
    def __init__(self, vision_config=None, text_config=None, **kwargs):
        super().__init__()
        if exists(vision_config):
            vision_config.update(kwargs)
        else:
            vision_config = kwargs

        if exists(text_config):
            text_config.update(kwargs)
        else:
            text_config = kwargs

        self.visual = FrozenOpenCLIPImageEmbedder(**vision_config)
        self.transformer = FrozenOpenCLIPEmbedder(**text_config)

    def preprocess(self, x):
        return self.visual.preprocess(x)

    @property
    def scale_factor(self):
        return self.visual.scale_factor

    def update_scale_factor(self, scale_factor):
        self.visual.update_scale_factor(scale_factor)

    def encode(self, *args, **kwargs):
        return self.visual.encode(*args, **kwargs)

    @torch.no_grad()
    def encode_text(self, text, normalize=True):
        return self.transformer(text, normalize)

    def calculate_scale(self, v: torch.Tensor, t: torch.Tensor):
        """
            Calculate the projection of v along the direction of t
            params:
                v: visual tokens from clip image encoder, shape: (b, n, c)
                t: text features from clip text encoder (argmax -1), shape: (b, 1, c)
        """
        return v @ t.mT



class HFCLIPVisionModel(nn.Module):
    # TODO: open_clip_torch is incompatible with deepspeed ZeRO3, change to huggingface implementation in the future
    def __init__(self, arch="ViT-bigG-14", image_size=224, scale_factor=1.):
        super().__init__()
        self.model = CLIPVisionModelWithProjection.from_pretrained(
            hf_versions[arch],
            cache_dir = cache_dir
        )
        self.image_size = image_size
        self.scale_factor = scale_factor
        self.register_buffer(
            'mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]).view(1, -1, 1, 1), persistent=False
        )
        self.register_buffer(
            'std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]).view(1, -1, 1, 1), persistent=False
        )
        self.antialias = True
        self.requires_grad_(False).eval()

    def preprocess(self, x):
        # normalize to [0,1]
        ns = int(self.image_size * self.scale_factor)
        x = F.interpolate(x, (ns, ns), mode="bicubic", align_corners=True, antialias=self.antialias)
        x = (x + 1.0) / 2.0

        # renormalize according to clip
        x = (x - self.mean) / self.std
        return x

    def forward(self, x, output_type):
        outputs = self.model(x).last_hidden_state
        if output_type == "cls":
            outputs = outputs[:, :1]
        elif output_type == "local":
            outputs = outputs[:, 1:]
        outputs = self.model.vision_model.post_layernorm(outputs)
        outputs = self.model.visual_projection(outputs)
        return outputs

    @torch.no_grad()
    def encode(self, img, output_type="full", preprocess=True, warp_p=0., **kwargs):
        img = self.preprocess(img) if preprocess else img

        if warp_p > 0.:
            rand = append_dims(torch.rand(img.shape[0], device=img.device, dtype=img.dtype), img.ndim)
            img = torch.where(torch.Tensor(rand > warp_p), img, tps_warp(img))
        return self(img, output_type)




class FrozenT5Embedder(nn.Module):
    """Uses the T5 transformer encoder for text"""

    def __init__(
        self, version="google/t5-v1_1-xxl", device="cuda", max_length=77, freeze=True
    ):  # others are google/t5-v1_1-xl and google/t5-v1_1-xxl
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(version, cache_dir=cache_dir)
        self.transformer = T5EncoderModel.from_pretrained(version, cache_dir=cache_dir)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device)
        with torch.autocast("cuda", enabled=False):
            outputs = self.transformer(input_ids=tokens)
        z = outputs.last_hidden_state
        return z

    @torch.no_grad()
    def encode(self, text):
        return self(text)


class HFCLIPTextEmbedder(nn.Module):
    def __init__(self, arch, freeze=True, device="cuda", max_length=77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(
            hf_versions[arch],
            cache_dir = cache_dir
        )
        self.model = CLIPTextModel.from_pretrained(
            hf_versions[arch],
            cache_dir = cache_dir
        )
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()

    def freeze(self):
        self.model = self.model.eval()

        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, text):
        if isinstance(text, torch.Tensor) and text.dtype == torch.long:
            # Input is already tokenized
            tokens = text
        else:
            # Need to tokenize text input
            batch_encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            tokens = batch_encoding["input_ids"].to(self.device)
            
        outputs = self.model(input_ids=tokens)
        z = outputs.last_hidden_state
        return z

    @torch.no_grad()
    def encode(self, text, normalize=False):
        outputs = self(text)
        if normalize:
            outputs = outputs / outputs.norm(dim=-1, keepdim=True)
        return outputs


class ScalarEmbedder(nn.Module):
    """embeds each dimension independently and concatenates them"""

    def __init__(self, embed_dim, out_dim):
        super().__init__()
        self.timestep = Timestep(embed_dim)
        self.embed_layer = nn.Sequential(
            nn.Linear(embed_dim, out_dim),
            nn.SiLU(),
            zero_module(nn.Linear(out_dim, out_features=out_dim))
        )

    def forward(self, x, dtype=torch.float32):
        emb = self.timestep(x)
        emb = rearrange(emb, "b d -> b 1 d")
        emb = self.embed_layer(emb.to(dtype))
        return emb


class TimestepEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.timestep = Timestep(embed_dim)

    def forward(self, x):
        x = self.timestep(x)
        return x


if __name__ == '__main__':
    import PIL.Image as Image

    encoder = FrozenOpenCLIPImageEmbedder(arch="DFN-ViT-H")
    image = Image.open("../../miniset/origin/70717450.jpg").convert("RGB")
    image = (torchvision.transforms.ToTensor()(image) - 0.5) * 2
    image = image.unsqueeze(0)
    print(image.shape)
    feat = encoder.encode(image, "local")
    print(feat.shape)