import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from refnet.modules.transformer import BasicTransformerBlock


def torch_dfs(model: nn.Module):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result


class AttentionMapVisualizer:
    """
    Enhanced visualizer for both self-attention and cross-attention maps.
    Supports token-level visualization with reference-side token selection.
    """
    def __init__(self, capture_mode='both'):
        """
        Args:
            capture_mode: 'self', 'cross', or 'both' - which attention to capture
        """
        self.self_attn_maps = {}
        self.cross_attn_maps = {}
        self.layer_names = []
        self.hooks = []
        self.enabled = False
        self.capture_mode = capture_mode
        
    def clear(self):
        """Clear all captured attention maps and hooks"""
        self.self_attn_maps = {}
        self.cross_attn_maps = {}
        self.layer_names = []
        self.clear_hooks()
    
    def clear_hooks(self):
        """Clear only the hooks without removing captured data"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def clear_data(self):
        """Clear only the captured data without removing hooks"""
        self.self_attn_maps = {}
        self.cross_attn_maps = {}
    
    def set_capture_mode(self, mode):
        """
        Set which attention type to capture
        Args:
            mode: 'self', 'cross', or 'both'
        """
        if mode not in ['self', 'cross', 'both']:
            raise ValueError(f"Invalid capture mode: {mode}. Must be 'self', 'cross', or 'both'")
        self.capture_mode = mode
        
    def register_hooks(self, model, layer_filter=None, clear_existing_data=True):
        """
        Register hooks to capture attention maps
        Args:
            model: The model to register hooks on
            layer_filter: Optional filter for layers
            clear_existing_data: If True, clear previously captured data; if False, keep existing data
        """
        self.clear_hooks()
        self.layer_names = []
        if clear_existing_data:
            self.clear_data()
        self.enabled = True
        
        def create_self_attn_hook(layer_name):
            def hook(module, input, output):
                if not self.enabled:
                    return
                # Capture attention map during forward pass
                if hasattr(module, '_attn_map'):
                    self.self_attn_maps[layer_name] = module._attn_map.detach().cpu()
            return hook
            
        def create_cross_attn_hook(layer_name):
            def hook(module, input, output):
                if not self.enabled:
                    return
                if hasattr(module, '_attn_map'):
                    self.cross_attn_maps[layer_name] = module._attn_map.detach().cpu()
            return hook
        
        # Hook attention layers in the model
        layer_idx = 0
        for name, module in model.named_modules():
            if 'attn1' in name and hasattr(module, 'forward'):
                # Self-attention - skip if not capturing
                if self.capture_mode not in ['self', 'both']:
                    continue
                    
                layer_name = f"{layer_idx}_self_{name}"
                self.layer_names.append(layer_name)
                hook = module.register_forward_hook(create_self_attn_hook(layer_name))
                self.hooks.append(hook)
                
                # Monkey patch the attention module to capture attention map
                self._patch_attention_module(module, is_self_attn=True)
                layer_idx += 1
                
            elif 'attn2' in name and hasattr(module, 'forward'):
                # Cross-attention - skip if not capturing
                if self.capture_mode not in ['cross', 'both']:
                    continue
                    
                layer_name = f"{layer_idx}_cross_{name}"
                self.layer_names.append(layer_name)
                hook = module.register_forward_hook(create_cross_attn_hook(layer_name))
                self.hooks.append(hook)
                
                # Monkey patch the attention module to capture attention map
                self._patch_attention_module(module, is_self_attn=False)
                layer_idx += 1
    
    def _patch_attention_module(self, module, is_self_attn=True):
        """Patch attention module to capture attention maps"""
        if not hasattr(module, 'attn_forward'):
            return
            
        original_attn_forward = module.attn_forward
        capture_mode = self.capture_mode
        
        def patched_attn_forward(q, k, v, scale=1., grid_size=None, mask=None):
            # Check if we should capture this attention type
            should_capture = (
                capture_mode == 'both' or
                (capture_mode == 'self' and is_self_attn) or
                (capture_mode == 'cross' and not is_self_attn)
            )
            
            if should_capture:
                # Compute attention map
                try:
                    # Apply normalization if available
                    if hasattr(module, 'q_norm') and hasattr(module, 'k_norm'):
                        q_normalized = module.q_norm(q)
                        k_normalized = module.k_norm(k)
                    else:
                        q_normalized = q
                        k_normalized = k
                    
                    # Apply rope if available
                    if hasattr(module, 'rope') and grid_size is not None:
                        q_reshaped = rearrange(q_normalized, "b n (h c) -> b n h c", h=module.heads)
                        k_reshaped = rearrange(k_normalized, "b n (h c) -> b n h c", h=module.heads)
                        q_normalized = module.rope(q_reshaped, grid_size)
                        k_normalized = module.rope(k_reshaped, grid_size)
                    else:
                        q_reshaped = rearrange(q_normalized, "b n (h c) -> b n h c", h=module.heads)
                        k_reshaped = rearrange(k_normalized, "b n (h c) -> b n h c", h=module.heads)
                        q_normalized = q_reshaped
                        k_normalized = k_reshaped
                    
                    attn_weights = torch.einsum('bnhc,bmhc->bhnm', q_normalized, k_normalized) / (q_normalized.shape[-1] ** 0.5)
                    attn_weights = F.softmax(attn_weights, dim=-1)
                    
                    module._attn_map = attn_weights.detach()
                except Exception as e:
                    print(f"Warning: Failed to capture attention map: {e}")
            
            # Call original forward
            return original_attn_forward(q, k, v, scale, grid_size, mask)
        
        module.attn_forward = patched_attn_forward
    
    def get_attention_map(self, layer_name, token_idx=None, is_cross_attn=True):
        attn_dict = self.cross_attn_maps if is_cross_attn else self.self_attn_maps
        
        if layer_name not in attn_dict:
            return None
            
        attn_map = attn_dict[layer_name]  # [batch, heads, n_query, n_kv]
        
        if token_idx is not None:
            print(f"token_size: {attn_map.size(2)}, token_idx: {token_idx}")
            attn_map = attn_map[:, :, token_idx, :]  # [batch, heads, n_kv]
        
        return attn_map[0]  # Return first batch
    
    def visualize_attention_on_image(self, attn_map, image, spatial_size=None, source_token_idx=None):
        """
        Visualize attention map overlaid on image
        Args:
            attn_map: Attention map [heads, h*w] or [h*w]
            image: Reference image as numpy array
            spatial_size: (height, width) of the feature map
            source_token_idx: Optional source token index to highlight with red box
        Returns:
            List of visualization images (one per head if multi-head)
        """
        if len(attn_map.shape) == 2:
            # Multi-head attention
            num_heads = attn_map.shape[0]
        else:
            # Single attention map
            attn_map = attn_map.unsqueeze(0)
            num_heads = 1
        
        # Infer spatial size if not provided
        if spatial_size is None:
            n_tokens = attn_map.shape[1]
            sqrt_n = np.sqrt(n_tokens)
            
            # Check if it's a perfect square
            if int(sqrt_n) ** 2 == n_tokens:
                spatial_size = (int(sqrt_n), int(sqrt_n))
            else:
                # Try to factor the number into height and width
                # Try common aspect ratios and factorizations
                found = False
                for h in range(int(sqrt_n), 0, -1):
                    if n_tokens % h == 0:
                        w = n_tokens // h
                        # Prefer dimensions that are closer to square
                        if not found or abs(h - w) < abs(spatial_size[0] - spatial_size[1]):
                            spatial_size = (h, w)
                            found = True
                            if h == w:
                                break
                
                if not found:
                    raise ValueError(f"Cannot infer spatial dimensions from {n_tokens} tokens. Please provide spatial_size explicitly.")
                
                print(f"Inferred spatial size {spatial_size} from {n_tokens} tokens")
        
        h, w = spatial_size
        
        # Validate that spatial size matches token count
        if h * w != attn_map.shape[1]:
            raise ValueError(f"Spatial size {spatial_size} (h*w={h*w}) does not match attention map size {attn_map.shape[1]}")
        img_h, img_w = image.shape[:2]
        
        results = []
        for head_idx in range(num_heads):
            # Reshape to spatial dimensions
            attn = attn_map[head_idx].reshape(h, w).numpy()
            
            # Normalize to [0, 255]
            attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
            attn = (attn * 255).astype(np.uint8)
            
            # Resize to image size
            attn_resized = cv2.resize(attn, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
            
            # Apply colormap
            heatmap = cv2.applyColorMap(attn_resized, cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # Overlay on image
            overlay = cv2.addWeighted(image, 0.4, heatmap, 0.6, 0)
            
            # Draw red box around source token if specified
            if source_token_idx is not None:
                if isinstance(source_token_idx, tuple):
                    grid_h, grid_w = source_token_idx
                else:
                    grid_h = source_token_idx // w
                    grid_w = source_token_idx % w
                
                # Calculate grid cell size in image coordinates
                cell_h = img_h / h
                cell_w = img_w / w
                
                # Calculate box coordinates
                x1 = int(grid_w * cell_w)
                y1 = int(grid_h * cell_h)
                x2 = int((grid_w + 1) * cell_w)
                y2 = int((grid_h + 1) * cell_h)
                
                # Draw red box
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 3)
            
            results.append(overlay)
        
        return results
    
    def get_layer_names(self, filter_type=None):
        if filter_type == 'self':
            return [name for name in self.layer_names if '_self_' in name]
        elif filter_type == 'cross':
            return [name for name in self.layer_names if '_cross_' in name]
        else:
            return self.layer_names
    
    def visualize_token_attention(self, token_idx, reference_image, is_cross_attn=True, 
                                  layer_names=None, aggregate='mean', spatial_size=None):
        if layer_names is None:
            layer_names = self.get_layer_names('cross' if is_cross_attn else 'self')
        
        results = {}
        for layer_name in layer_names:
            attn_map = self.get_attention_map(layer_name, token_idx, is_cross_attn)
            if attn_map is None:
                continue
            
            # Aggregate across heads if requested
            if aggregate == 'mean':
                attn_map = attn_map.mean(dim=0)
            elif aggregate == 'max':
                attn_map = attn_map.max(dim=0)[0]
            
            # Visualize with red box around source token
            vis_images = self.visualize_attention_on_image(
                attn_map, reference_image, 
                spatial_size=spatial_size,
                source_token_idx=token_idx
            )
            results[layer_name] = vis_images
        
        return results


class GramMatrixVisualizer:
    """
    Visualizer for Gram matrices computed from BasicTransformerBlock outputs.
    Similar to AttentionMapVisualizer but captures feature correlations.
    """
    def __init__(self, apply_dist_norm=True, apply_norm=True):
        self.gram_matrices = {}
        self.layer_names = []
        self.hooks = []
        self.enabled = False
        self.apply_dist_norm = apply_dist_norm
        self.apply_norm = apply_norm
        self.current_step = 0
        self.max_steps = 0
        
    def clear(self):
        """Clear all captured gram matrices and hooks"""
        self.gram_matrices = {}
        self.layer_names = []
        self.clear_hooks()
    
    def clear_hooks(self):
        """Clear only the hooks without removing captured data"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def clear_data(self):
        """Clear only the captured data without removing hooks"""
        self.gram_matrices = {}
        self.current_step = 0
        self.max_steps = 0
        
    def register_hooks(self, model, layer_indices=None, clear_existing_data=True):
        """
        Register hooks to capture transformer block outputs
        Args:
            model: The model to register hooks on
            layer_indices: List of transformer block indices to hook (None = hook all)
            clear_existing_data: If True, clear previously captured data; if False, keep existing data
        """
        self.clear_hooks()
        self.layer_names = []
        if clear_existing_data:
            self.clear_data()
        self.enabled = True
        
        def create_gram_hook(layer_name):
            def hook(module, input, output):
                if not self.enabled:
                    return
                # Compute and store gram matrix for current step
                feat = output.detach()
                gram = self._compute_gram_matrix(feat)
                
                # Initialize layer dict if not exists
                if layer_name not in self.gram_matrices:
                    self.gram_matrices[layer_name] = {}
                
                # Store gram matrix for current step
                self.gram_matrices[layer_name][self.current_step] = gram.cpu()
                
                # Update max steps
                self.max_steps = max(self.max_steps, self.current_step + 1)
            return hook
        
        # Find all transformer blocks
        layer_idx = 0
        all_blocks = []
        for module in torch_dfs(model):
            if isinstance(module, BasicTransformerBlock):
                all_blocks.append((layer_idx, module))
                layer_idx += 1
        
        # Filter by indices if specified
        if layer_indices is not None:
            all_blocks = [(idx, module) for idx, module in all_blocks if idx in layer_indices]
        
        # Register hooks
        for idx, module in all_blocks:
            layer_name = f"block_{idx}"
            self.layer_names.append(layer_name)
            hook = module.register_forward_hook(create_gram_hook(layer_name))
            self.hooks.append(hook)
        
        print(f"Registered gram matrix hooks on {len(self.layer_names)} transformer blocks")
    
    def _compute_gram_matrix(self, features):
        """
        Compute gram matrix from features
        Args:
            features: [batch, seq_len, channels]
        Returns:
            gram: [batch, seq_len, seq_len] gram matrix
        """
        if self.apply_dist_norm:
            features = (features - features.mean(dim=1, keepdim=True)) / (features.std(dim=1, keepdim=True) + 1e-6)
            
        if self.apply_norm:
            features = F.normalize(features, dim=-1)
        
        # Compute gram matrix: G = F * F^T
        gram = torch.matmul(features, features.transpose(-1, -2))
        return gram
    
    def get_gram_matrix(self, layer_name, step=-1):
        """
        Get gram matrix for a specific layer and step
        Args:
            layer_name: Name of the layer
            step: Denoising step (-1 = last step)
        """
        layer_data = self.gram_matrices.get(layer_name, None)
        if layer_data is None:
            return None
        
        if step == -1:
            # Get the last step
            if not layer_data:
                return None
            step = max(layer_data.keys())
        
        return layer_data.get(step, None)
    
    def increment_step(self):
        """Increment the current step counter"""
        self.current_step += 1
    
    def reset_step(self):
        """Reset step counter to 0"""
        self.current_step = 0
    
    def get_available_steps(self):
        """Get list of available denoising steps"""
        if not self.gram_matrices:
            return []
        # Get steps from first layer (all layers should have same steps)
        first_layer = next(iter(self.gram_matrices.values()))
        return sorted(first_layer.keys())
    
    def get_layer_names(self):
        """Get list of all captured layer names"""
        return self.layer_names
    
    def visualize_gram_similarity(self, gram_matrix, image, target_patch_idx, spatial_size, alpha=0.6):
        """
        Visualize similarity from a target patch to all other patches
        Args:
            gram_matrix: [seq_len, seq_len] or [batch, seq_len, seq_len]
            image: Image to overlay on (numpy array)
            target_patch_idx: Target patch index (h, w) tuple or linear index
            spatial_size: (h, w) spatial dimensions of feature map (target space)
            alpha: Overlay transparency
        Returns:
            Overlay image
        """
        if len(gram_matrix.shape) == 3:
            gram_matrix = gram_matrix[0]
        
        # Get actual sequence length from gram matrix
        actual_seq_len = gram_matrix.shape[0]
        
        # Get target spatial size to determine aspect ratio
        target_h, target_w = spatial_size
        aspect_ratio = target_w / target_h
        
        # Factor the sequence length to find best matching dimensions
        sqrt_n = np.sqrt(actual_seq_len)
        best_h, best_w = int(sqrt_n), int(sqrt_n)
        best_ratio_diff = float('inf')
        
        # Try all possible factorizations
        for h in range(1, int(sqrt_n) + 1):
            if actual_seq_len % h == 0:
                w = actual_seq_len // h
                # Prefer dimensions that match the target aspect ratio
                ratio_diff = abs(w / h - aspect_ratio)
                if ratio_diff < best_ratio_diff:
                    best_h, best_w = h, w
                    best_ratio_diff = ratio_diff
        
        actual_h, actual_w = best_h, best_w
        
        print(f"Gram matrix size: {gram_matrix.shape}, actual spatial: ({actual_h}, {actual_w}), seq_len: {actual_seq_len}")
        if isinstance(target_patch_idx, tuple):
            h_idx, w_idx = target_patch_idx
        else:
            h_idx = target_patch_idx // target_w
            w_idx = target_patch_idx % target_w
        
        # Scale to actual spatial dimensions
        scaled_h = int(h_idx * actual_h / target_h)
        scaled_w = int(w_idx * actual_w / target_w)
        scaled_h = min(max(0, scaled_h), actual_h - 1)
        scaled_w = min(max(0, scaled_w), actual_w - 1)
        target_patch_idx = scaled_h * actual_w + scaled_w
        
        print(f"Original patch: ({h_idx}, {w_idx}), scaled to: ({scaled_h}, {scaled_w}), linear index: {target_patch_idx}")
        
        # Extract similarity vector for target patch
        similarity = gram_matrix[target_patch_idx].numpy()
        
        # Reshape to spatial dimensions (use actual dimensions)
        similarity_map = similarity.reshape(actual_h, actual_w)
        
        # Normalize to [0, 1]
        sim_min = similarity_map.min()
        sim_max = similarity_map.max()
        if sim_max > sim_min:
            similarity_map = (similarity_map - sim_min) / (sim_max - sim_min)
        else:
            similarity_map = np.zeros_like(similarity_map)
        
        # Convert to [0, 255]
        similarity_uint8 = (similarity_map * 255).astype(np.uint8)
        
        # Resize to image size
        img_h, img_w = image.shape[:2]
        similarity_resized = cv2.resize(similarity_uint8, (img_w, img_h), 
                                       interpolation=cv2.INTER_CUBIC)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(similarity_resized, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay on image
        overlay = cv2.addWeighted(image, 1-alpha, heatmap, alpha, 0)
        
        # Mark target patch location (use scaled coordinates)
        patch_h_size = img_h / actual_h
        patch_w_size = img_w / actual_w
        center_y = int((scaled_h + 0.5) * patch_h_size)
        center_x = int((scaled_w + 0.5) * patch_w_size)
        
        # Draw marker
        cv2.circle(overlay, (center_x, center_y), radius=8, color=(255, 0, 0), thickness=-1)
        cv2.circle(overlay, (center_x, center_y), radius=11, color=(255, 255, 255), thickness=2)
        
        return overlay