import os

import torch.hub
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tf
import functools

# Preprocessor weights are auto-downloaded to this directory on first use.
model_path = "preprocessor/weights"
os.environ["HF_HOME"] = model_path
torch.hub.set_dir(model_path)

from torch.hub import download_url_to_file
from transformers import AutoModelForImageSegmentation
from .anime2sketch import UnetGenerator
from .manga_line_extractor import res_skip
from .sketchKeras import SketchKeras
from .sk_model import LineartDetector
from .anime_segment import ISNetDIS
from ckpt_util import load_weights


class NoneMaskExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.identity = nn.Identity()
    
    def proceed(self, x: torch.Tensor, th=None, tw=None, dilate=False, *args, **kwargs):
        b, c, h, w = x.shape
        return torch.zeros([b, 1, h, w], device=x.device)
    
    def forward(self, x):
        return self.proceed(x)


# Line extractor and mask extractor weights hosted on HuggingFace.
remote_model_dict = {
    "lineart": "https://huggingface.co/lllyasviel/Annotators/resolve/main/netG.pth",
    "lineart_denoise": "https://huggingface.co/lllyasviel/Annotators/resolve/main/erika.pth",
    "lineart_keras": "https://huggingface.co/tellurion/line_extractor/resolve/main/model.pth",
    "lineart_sk": "https://huggingface.co/lllyasviel/Annotators/resolve/main/sk_model.pth",
    "ISNet": "https://huggingface.co/tellurion/line_extractor/resolve/main/isnetis.safetensors",
    "ISNet-sketch": "https://huggingface.co/tellurion/line_extractor/resolve/main/sketch-segment.safetensors" 
}

# Background removal models loaded via HuggingFace transformers.
BiRefNet_dict = {
    "rmbg-v2": ("briaai/RMBG-2.0", 1024),
    "BiRefNet": ("ZhengPeng7/BiRefNet", 1024),
    "BiRefNet_HR": ("ZhengPeng7/BiRefNet_HR", 2048)
}

def rmbg_proceed(self, x: torch.Tensor, th=None, tw=None, dilate=False, *args, **kwargs):
    b, c, h, w = x.shape
    x = (x + 1.0) / 2.
    x = tf.resize(x, [self.image_size, self.image_size])
    x = tf.normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    x = self(x)[-1].sigmoid()
    x = tf.resize(x, [h, w])

    if th and tw:
        x = tf.pad(x, padding=[(th-h)//2, (tw-w)//2])
    if dilate:
        x = F.max_pool2d(x, kernel_size=21, stride=1, padding=10)
    x = torch.where(x > 0.5, torch.ones_like(x), torch.zeros_like(x))
    x = x.clamp(0, 1)
    return x



def create_model(model_name="lineart"):
    """Create a line extractor or mask extractor model by name.
    Weights are auto-downloaded from HuggingFace on first use.
    """
    if model_name == "none":
        return NoneMaskExtractor().eval()
    
    if model_name == "sam3":
        try:
            from .sam3_wrapper import SAM3Wrapper
            return SAM3Wrapper().eval()
        except ImportError as e:
            raise ImportError(
                f"Cannot load SAM3 model: {e}. "
                f"Please install SAM3 dependencies in preprocessor/sam3/ or use a different mask extractor."
            )

    if model_name in BiRefNet_dict.keys():
        model = AutoModelForImageSegmentation.from_pretrained(
            BiRefNet_dict[model_name][0],
            trust_remote_code = True,
            cache_dir = model_path,
            device_map = None,
            low_cpu_mem_usage = False,
        )
        model.eval()
        model.image_size = BiRefNet_dict[model_name][1]
        model.proceed = rmbg_proceed.__get__(model, model.__class__)
        return model

    assert model_name in remote_model_dict.keys()
    remote_path = remote_model_dict[model_name]
    basename = os.path.basename(remote_path)
    ckpt_path = os.path.join(model_path, basename)

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if not os.path.exists(ckpt_path):
        cache_path = "preprocessor/weights/weights.tmp"
        download_url_to_file(remote_path, dst=cache_path)
        os.rename(cache_path, ckpt_path)

    if model_name == "lineart":
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        model = UnetGenerator(3, 1, 8, 64, norm_layer=norm_layer, use_dropout=False)
    elif model_name == "lineart_denoise":
        model = res_skip()
    elif model_name == "lineart_keras":
        model = SketchKeras()
    elif model_name == "lineart_sk":
        model = LineartDetector()
    elif model_name == "ISNet" or model_name == "ISNet-sketch":
        model = ISNetDIS()
    else:
        return None

    ckpt = load_weights(ckpt_path)
    for key in list(ckpt.keys()):
        if 'module.' in key:
            ckpt[key.replace('module.', '')] = ckpt[key]
            del ckpt[key]
    model.load_state_dict(ckpt)
    return model.eval()