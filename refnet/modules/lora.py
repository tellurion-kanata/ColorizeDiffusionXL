import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, Dict, List
from einops import rearrange
from refnet.util import exists, default
from refnet.modules.transformer import BasicTransformerBlock, SelfInjectedTransformerBlock


def get_module_safe(self, module_path: str):
    current_module = self
    try:
        for part in module_path.split('.'):
            current_module = getattr(current_module, part)
        return current_module
    except AttributeError:
        raise AttributeError(f"Cannot find modules {module_path}")


def switch_lora(self, v, label=None):
    for t in [self.to_q, self.to_k, self.to_v]:
        t.set_lora_active(v, label)


def lora_forward(self, x, context, mask, scale=1., scale_factor= None):
    def qkv_forward(x, context):
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        return q, k, v

    assert exists(scale_factor), "Scale factor must be assigned before masked attention"

    mask = rearrange(
        F.interpolate(mask, scale_factor=scale_factor, mode="bicubic"),
        "b c h w -> b (h w) c"
    ).contiguous()

    c1, c2 = context.chunk(2, dim=1)

    # Background region cross-attention
    if self.use_lora:
        self.switch_lora(False, "foreground")
    q2, k2, v2 = qkv_forward(x, c2)
    bg_out = self.attn_forward(q2, k2, v2, scale) * self.bg_scale

    # Character region cross-attention
    if self.use_lora:
        self.switch_lora(True, "foreground")
    q1, k1, v1 = qkv_forward(x, c1)
    fg_out = self.attn_forward(q1, k1, v1, scale) * self.fg_scale

    fg_out = fg_out * (1 - self.merge_scale) + bg_out * self.merge_scale
    return fg_out * mask + bg_out * (1 - mask)
    # return torch.where(mask > self.mask_threshold, fg_out, bg_out)


def dual_lora_forward(self, x, context, mask, scale=1., scale_factor=None):
    """
    This function hacks cross-attention layers.
    Args:
        x: Query input
        context: Key and value input
        mask: Character mask
        scale: Attention scale
        sacle_factor: Current latent size factor

    """
    def qkv_forward(x, context):
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        return q, k, v

    assert exists(scale_factor), "Scale factor must be assigned before masked attention"

    mask = rearrange(
        F.interpolate(mask, scale_factor=scale_factor, mode="bicubic"),
        "b c h w -> b (h w) c"
    ).contiguous()

    c1, c2 = context.chunk(2, dim=1)

    # Background region cross-attention
    if self.use_lora:
        self.switch_lora(True, "background")
        self.switch_lora(False, "foreground")
    q2, k2, v2 = qkv_forward(x, c2)
    bg_out = self.attn_forward(q2, k2, v2, scale) * self.bg_scale

    # Foreground region cross-attention
    if self.use_lora:
        self.switch_lora(False, "background")
        self.switch_lora(True, "foreground")
    q1, k1, v1 = qkv_forward(x, c1)
    fg_out = self.attn_forward(q1, k1, v1, scale) * self.fg_scale

    fg_out = fg_out * (1 - self.merge_scale) + bg_out * self.merge_scale
    # return fg_out * mask + bg_out * (1 - mask)
    return torch.where(mask > self.mask_threshold, fg_out, bg_out)



class MultiLoraInjectedLinear(nn.Linear):
    """
    A linear layer that can hold multiple LoRA adapters and merge them.
    """
    def __init__(
            self,
            in_features,
            out_features,
            bias = False,
    ):
        super().__init__(in_features, out_features, bias)
        self.lora_adapters: Dict[str, Dict[str, nn.Module]] = {}  # {label: {up/down: layer}}
        self.lora_scales: Dict[str, float] = {}
        self.active_loras: Dict[str, bool] = {}
        self.original_weight = None
        self.original_bias = None
        
        # Freeze original weights
        self.weight.requires_grad_(False)
        if exists(self.bias):
            self.bias.requires_grad_(False)

    def add_lora_adapter(self, label: str, r: int, scale: float = 1.0, dropout_p: float = 0.0):
        """Add a new LoRA adapter with the given label."""
        if isinstance(r, float):
            r = int(r * self.out_features)
            
        lora_down = nn.Linear(self.in_features, r, bias=self.bias is not None)
        lora_up = nn.Linear(r, self.out_features, bias=self.bias is not None)
        dropout = nn.Dropout(dropout_p)
        
        # Initialize weights
        nn.init.normal_(lora_down.weight, std=1 / r)
        nn.init.zeros_(lora_up.weight)
        
        self.lora_adapters[label] = {
            'down': lora_down,
            'up': lora_up,
            'dropout': dropout,
        }
        self.lora_scales[label] = scale
        self.active_loras[label] = True
        
        # Register as submodules
        self.add_module(f'lora_down_{label}', lora_down)
        self.add_module(f'lora_up_{label}', lora_up)
        self.add_module(f'lora_dropout_{label}', dropout)

    def get_trainable_layers(self, label: str = None):
        """Get trainable layers for specific LoRA or all LoRAs."""
        layers = []
        if exists(label):
            if label in self.lora_adapters:
                adapter = self.lora_adapters[label]
                layers.extend([adapter['down'], adapter['up']])
        else:
            for adapter in self.lora_adapters.values():
                layers.extend([adapter['down'], adapter['up']])
        return layers

    def set_lora_active(self, active: bool, label: str):
        """Activate or deactivate a specific LoRA adapter."""
        if label in self.active_loras:
            self.active_loras[label] = active

    def set_lora_scale(self, scale: float, label: str):
        """Set the scale for a specific LoRA adapter."""
        if label in self.lora_scales:
            self.lora_scales[label] = scale

    def merge_lora_weights(self, labels: List[str] = None):
        """Merge specified LoRA adapters into the base weights."""
        if labels is None:
            labels = list(self.lora_adapters.keys())
        
        # Store original weights if not already stored
        if self.original_weight is None:
            self.original_weight = self.weight.clone()
            if exists(self.bias):
                self.original_bias = self.bias.clone()
        
        merged_weight = self.original_weight.clone()
        merged_bias = self.original_bias.clone() if exists(self.original_bias) else None
        
        for label in labels:
            if label in self.lora_adapters and self.active_loras.get(label, False):
                lora_up, lora_down = self.lora_adapters[label]['up'], self.lora_adapters[label]['down']
                scale = self.lora_scales[label]

                lora_weight = lora_up.weight @ lora_down.weight
                merged_weight += scale * lora_weight
                
                if exists(merged_bias) and exists(lora_up.bias):
                    lora_bias = lora_up.bias + lora_up.weight @ lora_down.bias
                    merged_bias += scale * lora_bias
        
        # Update weights
        self.weight = nn.Parameter(merged_weight, requires_grad=False)
        if exists(merged_bias):
            self.bias = nn.Parameter(merged_bias, requires_grad=False)
        
        # Deactivate all LoRAs after merging
        for label in labels:
            self.active_loras[label] = False

    def recover_original_weight(self):
        """Recover the original weights before any LoRA modifications."""
        if self.original_weight is not None:
            self.weight = nn.Parameter(self.original_weight.clone())
            if exists(self.original_bias):
                self.bias = nn.Parameter(self.original_bias.clone())
            
            # Reactivate all LoRAs
            for label in self.active_loras:
                self.active_loras[label] = True

    def forward(self, input):
        output = super().forward(input)
        
        # Add contributions from active LoRAs
        for label, adapter in self.lora_adapters.items():
            if self.active_loras.get(label, False):
                lora_out = adapter['up'](adapter['dropout'](adapter['down'](input)))
                output += self.lora_scales[label] * lora_out
                
        return output


class LoraModules:
    def __init__(self, sd, lora_params, *args, **kwargs):
        self.modules = {}
        self.multi_lora_layers: Dict[str, MultiLoraInjectedLinear] = {}  # path -> MultiLoraLayer

        for cfg in lora_params:
            root_module = get_module_safe(sd, cfg.pop("root_module"))
            label = cfg.pop("label", "lora")
            self.inject_lora(label, root_module, **cfg)

    def inject_lora(
            self,
            label,
            root_module,
            r,
            split_forward = False,
            target_keys = ("to_q", "to_k", "to_v"),
            filter_keys = None,
            target_class = None,
            scale = 1.0,
            dropout_p = 0.0,
    ):
        def check_condition(path, child, class_list):
            if exists(filter_keys) and any(path.find(key) > -1 for key in filter_keys):
                return False
            if exists(target_keys) and any(path.endswith(key) for key in target_keys):
                return True
            if exists(class_list) and any(
                    isinstance(child, module_class) for module_class in class_list
            ):
                return True
            return False

        def retrieve_target_modules():
            from refnet.util import get_obj_from_str
            target_class_list = [get_obj_from_str(t) for t in target_class] if exists(target_class) else None

            modules = []
            for name, module in root_module.named_modules():
                for key, child in module._modules.items():
                    full_path = name + '.' + key if name else key
                    if check_condition(full_path, child, target_class_list):
                        modules.append((module, child, key, full_path))
            return modules

        modules: list[Union[nn.Module]] = []
        retrieved_modules = retrieve_target_modules()

        for parent, child, child_name, full_path in retrieved_modules:
            # Check if this layer already has a MultiLoraInjectedLinear
            if full_path in self.multi_lora_layers:
                # Add LoRA to existing MultiLoraInjectedLinear
                multi_lora_layer = self.multi_lora_layers[full_path]
                multi_lora_layer.add_lora_adapter(label, r, scale, dropout_p)
            else:
                # Check if the current layer is already a MultiLoraInjectedLinear
                if isinstance(child, MultiLoraInjectedLinear):
                    child.add_lora_adapter(label, r, scale, dropout_p)
                    self.multi_lora_layers[full_path] = child
                else:
                    # Replace with MultiLoraInjectedLinear and add first LoRA
                    multi_lora_layer = MultiLoraInjectedLinear(
                        in_features=child.weight.shape[1],
                        out_features=child.weight.shape[0],
                        bias=exists(child.bias),
                    )

                    multi_lora_layer.add_lora_adapter(label, r, scale, dropout_p)
                    parent._modules[child_name] = multi_lora_layer
                    self.multi_lora_layers[full_path] = multi_lora_layer

            if split_forward:
                parent.masked_forward = dual_lora_forward.__get__(parent, parent.__class__)
            else:
                parent.masked_forward = lora_forward.__get__(parent, parent.__class__)

            parent.use_lora = True
            parent.switch_lora = switch_lora.__get__(parent, parent.__class__)
            modules.append(parent)

        self.modules[label] = modules
        print(f"Activated {label} lora with {len(self.multi_lora_layers)} layers")
        return self.multi_lora_layers, modules

    def get_trainable_layers(self, label = None):
        """Get all trainable layers, optionally filtered by label."""
        layers = []
        for lora_layer in self.multi_lora_layers.values():
            layers += lora_layer.get_trainable_layers(label)
        return layers

    def switch_lora(self, mode, label = None):
        if exists(label):
            for layer in self.multi_lora_layers.values():
                layer.set_lora_active(mode, label)
            for module in self.modules[label]:
                module.use_lora = mode
        else:
            for layer in self.multi_lora_layers.values():
                for lora_label in layer.lora_adapters.keys():
                    layer.set_lora_active(mode, lora_label)

            for modules in self.modules.values():
                for module in modules:
                    module.use_lora = mode

    def adjust_lora_scales(self, scale, label = None):
        if exists(label):
            for layer in self.multi_lora_layers.values():
                layer.set_lora_scale(scale, label)
        else:
            for layer in self.multi_lora_layers.values():
                for lora_label in layer.lora_adapters.keys():
                    layer.set_lora_scale(scale, lora_label)

    def merge_lora(self, labels = None):
        if labels is None:
            labels = list(self.modules.keys())
        elif isinstance(labels, str):
            labels = [labels]

        for layer in self.multi_lora_layers.values():
            layer.merge_lora_weights(labels)

    def recover_lora(self):
        for layer in self.multi_lora_layers.values():
            layer.recover_original_weight()

    def get_lora_info(self):
        """Get information about all LoRA adapters."""
        info = {}
        for path, layer in self.multi_lora_layers.items():
            info[path] = {
                'labels': list(layer.lora_adapters.keys()),
                'active': {label: active for label, active in layer.active_loras.items()},
                'scales': layer.lora_scales.copy()
            }
        return info