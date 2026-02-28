import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image


class SAM3Wrapper(nn.Module):
    def __init__(self, prompt="person"):
        super().__init__()
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        
        self.model = build_sam3_image_model()
        self.processor = Sam3Processor(self.model)
        self.default_prompt = prompt
        self.device_type = "cuda"
    
    def proceed(self, x: torch.Tensor, th=None, tw=None, dilate=False, pil_x=None, threshold=0.5, *args, **kwargs):
        b, c, h, w = x.shape
        device = x.device
        
        if pil_x is None:
            x_np = ((x[0].cpu().permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            pil_x = Image.fromarray(x_np)
        
        masks_list = []
        for i in range(b):
            if i > 0:
                x_np = ((x[i].cpu().permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                pil_x = Image.fromarray(x_np)
            
            state = self.processor.set_image(pil_x)
            output = self.processor.set_text_prompt(
                prompt=self.default_prompt,
                state=state
            )
            
            masks = output["masks"]
            scores = output["scores"]
            
            if len(masks) > 0:
                best_idx = scores.argmax()
                mask = masks[best_idx]
                
                if isinstance(mask, np.ndarray):
                    mask_tensor = torch.from_numpy(mask).float()
                else:
                    mask_tensor = mask.float()
                
                if mask_tensor.dim() == 2:
                    mask_tensor = mask_tensor.unsqueeze(0)
                
                if mask_tensor.shape[1:] != (h, w):
                    mask_tensor = F.interpolate(
                        mask_tensor.unsqueeze(0), 
                        size=(h, w), 
                        mode='bilinear', 
                        align_corners=False
                    ).squeeze(0)
            else:
                mask_tensor = torch.zeros(1, h, w)
            
            masks_list.append(mask_tensor)
        
        result = torch.stack(masks_list, dim=0).to(device)
        
        if th and tw:
            pad_h = (th - h) // 2
            pad_w = (tw - w) // 2
            result = F.pad(result, (pad_w, pad_w, pad_h, pad_h))
        
        if dilate:
            result = F.max_pool2d(result, kernel_size=21, stride=1, padding=10)
        
        result = result.clamp(0, 1)
        return result
    
    def forward(self, x):
        return self.proceed(x)
    
    def cuda(self):
        self.model = self.model.cuda()
        self.processor.device = "cuda"
        self.device_type = "cuda"
        return self
    
    def cpu(self):
        self.model = self.model.cpu()
        self.processor.device = "cpu"
        self.device_type = "cpu"
        return self
    
    def eval(self):
        self.model.eval()
        return self




