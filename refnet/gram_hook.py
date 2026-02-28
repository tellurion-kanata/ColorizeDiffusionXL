import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .modules.transformer import BasicTransformerBlock


def torch_dfs(model: nn.Module):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result


class GramHooker:
    def __init__(self, apply_norm=True, apply_dist_norm=True, gram_ids=[23, 63], gram_type="ff", tau=1, kl_temp=1.0):
        assert gram_type in ["ff", "crossattn", "selfattn"]
        self.registered = False
        self.hooks = []
        self.feat_grams = []
        self.offload = False
        self.apply_norm = apply_norm
        self.apply_dist_norm = apply_dist_norm
        
        self.tau = tau
        self.kl_temp = kl_temp
        self.gram_ids = gram_ids
        self.hook_target = gram_type

    def register(self, unet):
        def attn_hooker(module, input, output):
            self.feat_grams.append(self.gram_matrix(output))
        transformer_idx = 0
        for module in torch_dfs(unet):
            if isinstance(module, BasicTransformerBlock) and hasattr(module, "attn2"):
                transformer_idx += 1
                if transformer_idx in self.gram_ids:
                    if self.hook_target == "ff":
                        hook = module.register_forward_hook(attn_hooker)
                    elif self.hook_target == "selfattn":
                        hook = module.attn1.register_forward_hook(attn_hooker)
                    elif self.hook_target == "crossattn":
                        hook = module.attn2.register_forward_hook(attn_hooker)
                    else:
                        raise ValueError(f"Unknown hook type {self.hook_target}")
                    self.hooks.append(hook)
                if transformer_idx > max(self.gram_ids):
                    break
        self.registered = True

    def gram_matrix(self, t: torch.Tensor):
        if self.apply_dist_norm:
            t = (t - t.mean(dim=1, keepdim=True)) / (t.std(dim=1, keepdim=True) + 1e-6)
        
        if self.apply_norm:
            t = F.normalize(t, dim=-1)
        return torch.matmul(t, t.transpose(-1, -2))

    def get_hidden_grams(self):
        return self.feat_grams

    def clean(self):
        self.feat_grams = []

    def remove_hook(self):
        self.feat_grams = []
        for hook in self.hooks:
            hook.remove()
        self.registered = False


class GramLoss(nn.Module):
    def __init__(
        self, 
        weight = 1.0, 
        mask_weight = False, 
        loss_fn = "mse_loss", 
        kl_temp = 1.0, 
        ranking_margin = 0.1,
        l2_norm = False,
    ):
        super().__init__()
        self.weight = weight
        self.mask_weight = mask_weight
        self.kl_temp = kl_temp
        self.l2_norm = l2_norm
        self.loss_fn = getattr(self, loss_fn)

    def normalize_mask(self, mask, feat, mask_threshold):
        scale_factor = (feat.size(1) / (mask.size(2) * mask.size(3))) ** 0.5
        mask = rearrange(
            F.interpolate(mask, scale_factor=scale_factor, mode="nearest"),
            "b c h w -> b (h w) c"
        )
        mask = torch.where(mask > mask_threshold, torch.ones_like(mask), torch.zeros_like(mask))
        return mask

    def mse_loss(self, feat, anchor_feat, mask=None):
        loss = 0
        if self.mask_weight:
            for gram_t, gram_a in zip(feat, anchor_feat):
                mask = mask @ mask.transpose(1, 2).contiguous()
                loss += (F.mse_loss(
                    gram_t.float(), gram_a.to(gram_t.device).float().detach(), reduction="none"
                ) * mask).mean()
        else:
            for gram_t, gram_a in zip(feat, anchor_feat):
                loss += F.mse_loss(
                    gram_t.float(), gram_a.to(gram_t.device).float().detach()
                )
        return loss

    def kl_loss(self, feat, anchor_feat, *args, **kwargs):
        loss = 0
        for gram_t, gram_a in zip(feat, anchor_feat):
            gram_a = gram_a.to(gram_t.device).float().detach()
            
            log_prob_t = F.log_softmax(gram_t / self.kl_temp, dim=-1)
            prob_a = F.softmax(gram_a / self.kl_temp, dim=-1)
            
            loss += F.kl_div(log_prob_t, prob_a, reduction='batchmean')
        return loss
    
    def ranking_loss(self, feat, anchor_feat, margin=0.1, alpha=1.0, **kwargs):
        loss = 0
        for gram_t, gram_a in zip(feat, anchor_feat):
            row_mean = gram_a.mean(dim=-1, keepdim=True)
            row_std = gram_a.std(dim=-1, keepdim=True)
            threshold = row_mean + alpha * row_std
            
            pos_mask = (gram_a > threshold).float()
            neg_mask = 1.0 - pos_mask
                
            # num_pos = pos_mask.sum(dim=-1, keepdim=True).clamp(min=1)
            # num_neg = neg_mask.sum(dim=-1, keepdim=True).clamp(min=1)

            t_pos_min = (gram_t * pos_mask + (1 - pos_mask) * 100.0).min(dim=-1, keepdim=True)[0]
            t_neg_max = (gram_t * neg_mask - (1 - neg_mask) * 100.0).max(dim=-1, keepdim=True)[0]

            # for calculate the mean of positive and negative samples
            # t_pos_mean = (target_gram * pos_mask).sum(dim=-1, keepdim=True) / num_pos
            # t_neg_mean = (target_gram * neg_mask).sum(dim=-1, keepdim=True) / num_neg

            loss += F.relu(t_neg_max - t_pos_min + margin).mean()
        return loss
    
    def bce_loss(self, feat, anchor_feat, alpha=1.0, mask=None, *args, **kwargs):
        def correlation_matrix(x):
            row_mean = x.mean(dim=-1, keepdim=True)
            row_std = x.std(dim=-1, keepdim=True)
            threshold = row_mean + alpha * row_std
            return (x > threshold).float()

        loss = 0
        for gram_t, gram_a in zip(feat, anchor_feat):
            gram_a = gram_a.to(gram_t.device).float().detach()
            
            target_mask = correlation_matrix(gram_a)
            pred_hard = correlation_matrix(gram_t)
            pred_soft = torch.sigmoid(gram_t)
            pred = pred_hard - pred_soft.detach() + pred_soft
            
            if self.mask_weight:
                weight_mask = mask @ torch.ones_like(mask.transpose(1, 2))
                loss += (F.binary_cross_entropy(pred, target_mask, reduction="none") * weight_mask).mean()
            else:
                loss += F.binary_cross_entropy(pred, target_mask)
        return loss

    def forward(self, feat, anchor_feat, mask=None, mask_threshold=0.05, *args, **kwargs):
        if self.l2_norm:
            feat = [F.normalize(t, dim=-1) for t in feat]
            anchor_feat = [F.normalize(t, dim=-1).detach() for t in anchor_feat]
        else:
            anchor_feat = [t.detach() for t in anchor_feat]
        mask = self.normalize_mask(mask, feat[0], mask_threshold)
        return self.loss_fn(feat, anchor_feat, mask=mask) * self.weight