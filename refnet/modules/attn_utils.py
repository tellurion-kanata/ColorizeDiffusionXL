import torch
import torch.nn.functional as F

ATTN_PRECISION = torch.float16

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
    FLASH_ATTN_AVAILABLE = False

except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False
    try:
        import flash_attn
        FLASH_ATTN_AVAILABLE = True
    except ModuleNotFoundError:
        FLASH_ATTN_AVAILABLE = False

try:
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False


def half(x):
    if x.dtype not in [torch.float16, torch.bfloat16]:
        x = x.to(ATTN_PRECISION)
    return x

def attn_processor(q, k, v, attn_mask = None, *args, **kwargs):
    if attn_mask is not None:
        if XFORMERS_IS_AVAILBLE:
            out = xformers.ops.memory_efficient_attention(
                q, k, v, attn_bias=attn_mask, *args, **kwargs
            )
        else:
            q, k, v = map(lambda t: t.transpose(1, 2), (q, k, v))
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, *args, **kwargs
            ).transpose(1, 2)
    else:
        if FLASH_ATTN_3_AVAILABLE:
            dtype = v.dtype
            q, k, v = map(lambda t: half(t), (q, k, v))
            out = flash_attn_interface.flash_attn_func(q, k, v, *args, **kwargs)[0].to(dtype)
        elif FLASH_ATTN_AVAILABLE:
            dtype = v.dtype
            q, k, v = map(lambda t: half(t), (q, k, v))
            out = flash_attn.flash_attn_func(q, k, v, *args, **kwargs).to(dtype)
        elif XFORMERS_IS_AVAILBLE:
            out = xformers.ops.memory_efficient_attention(q, k, v, *args, **kwargs)
        else:
            q, k, v = map(lambda t: t.transpose(1, 2), (q, k, v))
            out = F.scaled_dot_product_attention(q, k, v, *args, **kwargs).transpose(1, 2)
    return out


def flash_attn_varlen_func(q, k, v, **kwargs):
    if FLASH_ATTN_3_AVAILABLE:
        return flash_attn_interface.flash_attn_varlen_func(q, k, v, **kwargs)[0]
    else:
        return flash_attn.flash_attn_varlen_func(q, k, v, **kwargs)


def split_tensor_by_mask(tensor: torch.Tensor, mask: torch.Tensor, threshold: float = 0.5):
    """
    Split input tensor into foreground and background based on mask, then concatenate them.
    
    Args:
        tensor: Input tensor of shape (batch, seq_len, dim)
        mask: Binary mask of shape (batch, seq_len, 1) or (batch, seq_len)
        threshold: Threshold for mask binarization
        
    Returns:
        split_tensor: Concatenated tensor with foreground first, then background
        fg_indices: Indices of foreground elements for restoration
        bg_indices: Indices of background elements for restoration
        original_shape: Original tensor shape for restoration
    """
    batch_size, seq_len, *dims = tensor.shape
    device, dtype = tensor.device, tensor.dtype
    
    # Ensure mask has correct shape and binarize
    if mask.dim() == 2:
        mask = mask.unsqueeze(-1)
    binary_mask = (mask > threshold).squeeze(-1)  # Shape: (batch, seq_len)
    
    # Store indices for restoration (keep minimal loop for complex indexing)
    fg_indices = [torch.where(binary_mask[b])[0] for b in range(batch_size)]
    bg_indices = [torch.where(~binary_mask[b])[0] for b in range(batch_size)]
    
    # Count elements efficiently
    fg_counts = binary_mask.sum(dim=1)
    bg_counts = (~binary_mask).sum(dim=1) 
    max_fg_len = fg_counts.max().item()
    max_bg_len = bg_counts.max().item()
    
    # Early exit if no elements
    if max_fg_len == 0 and max_bg_len == 0:
        return torch.zeros(batch_size, 0, *dims, device=device, dtype=dtype), fg_indices, bg_indices, tensor.shape
    
    # Create output tensor
    split_tensor = torch.zeros(batch_size, max_fg_len + max_bg_len, *dims, device=device, dtype=dtype)
    
    # Vectorized approach using gather for better efficiency
    for b in range(batch_size):
        if len(fg_indices[b]) > 0:
            split_tensor[b, :len(fg_indices[b])] = tensor[b][fg_indices[b]]
        if len(bg_indices[b]) > 0:
            split_tensor[b, max_fg_len:max_fg_len + len(bg_indices[b])] = tensor[b][bg_indices[b]]
    
    return split_tensor, fg_indices, bg_indices, tensor.shape


def restore_tensor_from_split(split_tensor: torch.Tensor, fg_indices: list, bg_indices: list, 
                            original_shape: torch.Size):
    """
    Restore original tensor from split tensor using stored indices.
    
    Args:
        split_tensor: Split tensor from split_tensor_by_mask
        fg_indices: List of foreground indices for each batch
        bg_indices: List of background indices for each batch  
        original_shape: Original tensor shape
        
    Returns:
        restored_tensor: Restored tensor with original shape and ordering
    """
    batch_size, seq_len = original_shape[:2]
    dims = original_shape[2:]
    device, dtype = split_tensor.device, split_tensor.dtype
    
    # Calculate split point efficiently
    max_fg_len = max((len(fg) for fg in fg_indices), default=0)
    
    # Initialize restored tensor
    restored_tensor = torch.zeros(batch_size, seq_len, *dims, device=device, dtype=dtype)
    
    # Early exit if no elements to restore
    if split_tensor.shape[1] == 0:
        return restored_tensor
    
    # Split tensor parts
    fg_part = split_tensor[:, :max_fg_len] if max_fg_len > 0 else None
    bg_part = split_tensor[:, max_fg_len:] if split_tensor.shape[1] > max_fg_len else None
    
    # Restore in single loop with efficient indexing
    for b in range(batch_size):
        if fg_part is not None and len(fg_indices[b]) > 0:
            restored_tensor[b, fg_indices[b]] = fg_part[b, :len(fg_indices[b])]
        if bg_part is not None and len(bg_indices[b]) > 0:
            restored_tensor[b, bg_indices[b]] = bg_part[b, :len(bg_indices[b])]
    
    return restored_tensor
