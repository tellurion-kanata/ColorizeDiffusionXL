"""Gradio UI backend for ColorizeDiffusion XL.
Handles model loading, inference dispatch, visualization, and UI event callbacks.
"""

import os
import random
import traceback
import gradio as gr
import os.path as osp
import json

from datetime import datetime
from glob import glob

from ckpt_util import load_config
from refnet.util import instantiate_from_config
from refnet.visualizer import AttentionMapVisualizer, GramMatrixVisualizer
from preprocessor import create_model
from .functool import *

model = None
attn_visualizer = AttentionMapVisualizer()
gram_visualizer = GramMatrixVisualizer(apply_norm=True)

model_type = ""
model_path = "models"
current_checkpoint = ""
attn_vispath = "visualization"
default_line_extractor = "lineart_keras"
default_mask_extractor = "rmbg-v2"
global_seed = None

smask_extractor = create_model("ISNet-sketch").cpu()

MAXM_INT32 = 429496729
# Model type is determined by checkpoint filename prefix, matched against this list.
# Corresponding inference config is loaded from configs/inference/{model_type}.yaml.
# - sdxl: base model (configs/inference/sdxl.yaml -> InferenceWrapper)
# - xlv2: adapter model with ReferenceNet (configs/inference/xlv2.yaml -> InferenceWrapperXL)
model_types = ["sdxl", "xlv2"]

'''
    Gradio UI functions
'''


def switch_to_fp16():
    global model
    model.switch_to_fp16()
    gr.Info("Switch unet to half precision")


def switch_vae_to_fp16():
    global model_type
    model.switch_vae_to_fp16()
    gr.Info("Switch vae to half precision")


def switch_to_fp32():
    global model
    model.switch_to_fp32()
    gr.Info("Switch unet to full precision")


def switch_vae_to_fp32():
    global model_type
    model.switch_vae_to_fp32()
    gr.Info("Switch vae to full precision")


def get_checkpoints():
    ckpt_fmts = ["safetensors", "pth", "ckpt", "pt"]
    checkpoints = sorted([
        osp.basename(file) for ext in ckpt_fmts
        for file in glob(osp.join(model_path, f"*.{ext}"))
    ])
    return checkpoints


def update_models():
    global current_checkpoint
    checkpoints = get_checkpoints()
    if not checkpoints:
        return gr.update(choices=[], value=None)
    if current_checkpoint not in checkpoints:
        current_checkpoint = checkpoints[0]
    return gr.update(choices=checkpoints, value=current_checkpoint)


def switch_extractor(type):
    global line_extractor
    try:
        line_extractor = create_model(type)
        gr.Info(f"Switched to {type} extractor")
    except Exception as e:
        print(f"Error info: {e}") 
        print(traceback.print_exc())
        gr.Info(f"Failed in loading {type} extractor")


def switch_mask_extractor(type):
    global mask_extractor
    try:
        mask_extractor = create_model(type)
        gr.Info(f"Switched to {type} extractor")
    except Exception as e:
        print(f"Error info: {e}")
        print(traceback.print_exc())
        gr.Info(f"Failed in loading {type} extractor")


def apppend_prompt(target, anchor, control, scale, enhance, ts0, ts1, ts2, ts3, prompt):
    target = target.strip()
    anchor = anchor.strip()
    control = control.strip()
    if target == "": target = "none"
    if anchor == "": anchor = "none"
    if control == "": control = "none"
    new_p = (f"\n[target] {target}; [anchor] {anchor}; [control] {control}; [scale] {str(scale)}; "
             f"[enhanced] {str(enhance)}; [ts0] {str(ts0)}; [ts1] {str(ts1)}; [ts2] {str(ts2)}; [ts3] {str(ts3)}")
    return "", "", "", 0.0, False, 0.5, 0.55, 0.65, 0.95, (prompt + new_p).strip()


def clear_prompts():
    return ""


def load_model(ckpt_path):
    global model, model_type, current_checkpoint
    config_root = "configs/inference"

    try:
        new_model_type = model_type
        for key in model_types:
            if ckpt_path.startswith(key):
                new_model_type = key
                break

        if model_type != new_model_type or not "model" in globals():
            if "model" in globals() and exists(model):
                del model
            config_path = osp.join(config_root, f"{new_model_type}.yaml")
            new_model = instantiate_from_config(load_config(config_path).model).cpu().eval()
            print(f"Swithced to {new_model_type} model, loading weights from [{ckpt_path}]...")
            model = new_model

        model.parameterization = "eps" if ckpt_path.find("eps") > -1 else "v"
        model.init_from_ckpt(osp.join(model_path, ckpt_path), logging=True)
        model.switch_to_fp16()

        model_type = new_model_type
        current_checkpoint = ckpt_path
        print(f"Loaded model from [{ckpt_path}], model_type [{model_type}].")
        gr.Info("Loaded model successfully.")

    except Exception as e:
        print(f"Error type: {e}")
        print(traceback.print_exc())
        gr.Info("Failed in loading model.")


def get_last_seed():
    return global_seed or -1


def reset_random_seed():
    return -1


def save_results(sketch_img, reference_img, results_gallery, colorization_rating=5, sketch_fidelity_rating=5, reference_similarity_rating=5):
    """
    Save the sketch, reference image, and results to modality-based directories
    """
    try:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create modality-based directories
        base_dir = "saved_results"
        result_dir = osp.join(base_dir, "result")
        reference_dir = osp.join(base_dir, "reference")
        sketch_dir = osp.join(base_dir, "sketch")
        feedback_dir = osp.join(base_dir, "feedback")
        
        for dir_path in [result_dir, reference_dir, sketch_dir, feedback_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        metadata = {
            "timestamp": current_time,
            "ratings": {
                "colorization_effect": colorization_rating,
                "sketch_fidelity": sketch_fidelity_rating,
                "reference_similarity": reference_similarity_rating
            },
            "model_type": model_type,
            "checkpoint": current_checkpoint
        }
        
        # Save sketch - use processed sketch from results_gallery if available, otherwise use original
        sketch_to_save = None
        if exists(results_gallery) and len(results_gallery) >= 2:
            # Use the second-to-last image from results_gallery (processed sketch with correct dimensions)
            processed_sketch = results_gallery[-2]
            if processed_sketch is not None:
                # Handle both PIL Image objects and tuples
                if hasattr(processed_sketch, 'save'):
                    sketch_to_save = processed_sketch
                elif isinstance(processed_sketch, tuple) and len(processed_sketch) > 0:
                    sketch_to_save = processed_sketch[0] if hasattr(processed_sketch[0], 'save') else None
                
                # Convert from [-1, 1] range and invert colors (black background to white background)
                if sketch_to_save is not None:
                    # Convert PIL to numpy array
                    sketch_array = np.array(sketch_to_save)
                    
                    # If it's in [0, 255] range, convert to [-1, 1] first
                    if sketch_array.max() > 1:
                        sketch_array = (sketch_array / 127.5) - 1.0
                    
                    # Convert from [-1, 1] to [0, 1]
                    sketch_array = (sketch_array + 1.0) / 2.0
                    
                    # Invert colors: black background (0) -> white background (1), white lines (1) -> black lines (0)
                    sketch_array = 1.0 - sketch_array
                    
                    # Convert back to [0, 255] range
                    sketch_array = (sketch_array * 255).astype(np.uint8)
                    
                    # Create PIL Image
                    sketch_to_save = Image.fromarray(sketch_array)
        
        # Fallback to original sketch if processed sketch is not available
        if sketch_to_save is None:
            sketch_to_save = sketch_img
            
        if sketch_to_save is not None:
            sketch_path = osp.join(sketch_dir, f"{current_time}.png")
            sketch_to_save.save(sketch_path)
            metadata["sketch_saved"] = True
            metadata["sketch_path"] = sketch_path
        else:
            metadata["sketch_saved"] = False
            
        # Save reference if provided
        if reference_img is not None:
            reference_path = osp.join(reference_dir, f"{current_time}.png")
            reference_img.save(reference_path)
            metadata["reference_saved"] = True
            metadata["reference_path"] = reference_path
        else:
            metadata["reference_saved"] = False
            
        # Save results if provided
        if exists(results_gallery) and len(results_gallery) > 0:
            saved_count = 0
            saved_paths = []
            for i, result in enumerate(results_gallery[:-2]):
                if result is not None:
                    # Handle both PIL Image objects and tuples
                    if hasattr(result, 'save'):
                        # It's a PIL Image
                        img_obj = result
                    elif isinstance(result, tuple) and len(result) > 0:
                        # It's a tuple, get the first element (likely the image)
                        img_obj = result[0] if hasattr(result[0], 'save') else None
                    else:
                        img_obj = None
                    
                    if img_obj is not None and hasattr(img_obj, 'save'):
                        if i == 0:
                            # Main result
                            result_path = osp.join(result_dir, f"{current_time}.png")
                            img_obj.save(result_path)
                        else:
                            # Additional results
                            result_path = osp.join(result_dir, f"{current_time}_{i}.png")
                            img_obj.save(result_path)
                        saved_paths.append(result_path)
                        saved_count += 1
            metadata["results_saved"] = saved_count
            metadata["result_paths"] = saved_paths
        else:
            metadata["results_saved"] = 0
            
        # Save metadata to feedback directory
        metadata_path = osp.join(feedback_dir, f"{current_time}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        gr.Info(f"Results saved with timestamp: {current_time}")
        return f"✅ Saved {current_time}"
        
    except Exception as e:
        error_msg = f"Error saving results: {str(e)}"
        gr.Warning(error_msg)
        return f"❌ Save failed: {str(e)}"


def delete_last_result():
    """
    Delete the most recently saved result files from all directories
    """
    try:
        base_dir = "saved_results"
        result_dir = osp.join(base_dir, "result")
        reference_dir = osp.join(base_dir, "reference")
        sketch_dir = osp.join(base_dir, "sketch")
        feedback_dir = osp.join(base_dir, "feedback")
        
        # Check if directories exist
        if not osp.exists(feedback_dir):
            return "❌ No saved results found"
        
        # Find the most recent timestamp by looking at feedback files
        feedback_files = glob(osp.join(feedback_dir, "*.json"))
        if not feedback_files:
            return "❌ No saved results found"
        
        # Get the most recent file based on filename (timestamp)
        latest_file = max(feedback_files, key=lambda x: osp.basename(x))
        latest_timestamp = osp.splitext(osp.basename(latest_file))[0]
        
        deleted_files = []
        
        # Delete feedback file
        if osp.exists(latest_file):
            os.remove(latest_file)
            deleted_files.append(f"feedback/{latest_timestamp}.json")
        
        # Delete sketch file
        sketch_file = osp.join(sketch_dir, f"{latest_timestamp}.png")
        if osp.exists(sketch_file):
            os.remove(sketch_file)
            deleted_files.append(f"sketch/{latest_timestamp}.png")
        
        # Delete reference file
        reference_file = osp.join(reference_dir, f"{latest_timestamp}.png")
        if osp.exists(reference_file):
            os.remove(reference_file)
            deleted_files.append(f"reference/{latest_timestamp}.png")
        
        # Delete result files (main and additional results)
        result_pattern = osp.join(result_dir, f"{latest_timestamp}*.png")
        result_files = glob(result_pattern)
        for result_file in result_files:
            os.remove(result_file)
            deleted_files.append(f"result/{osp.basename(result_file)}")
        
        if deleted_files:
            gr.Info(f"Deleted result: {latest_timestamp}")
            return f"🗑️ Deleted {latest_timestamp}"
        else:
            return "❌ No files to delete"
            
    except Exception as e:
        error_msg = f"Error deleting results: {str(e)}"
        gr.Warning(error_msg)
        return f"❌ Delete failed: {str(e)}"


def visualize(reference, text, *args):
    return visualize_heatmaps(model, reference, parse_prompts(text), *args)


def enable_attention_visualization():
    """Enable attention map visualization by registering hooks"""
    global attn_visualizer, model
    if model is not None:
        attn_visualizer.register_hooks(model.model, clear_existing_data=False)
        gr.Info("Attention visualization enabled")
        return "✅ Enabled"
    else:
        gr.Warning("Please load a model first")
        return "❌ No model loaded"


def disable_attention_visualization():
    """Disable attention map visualization"""
    global attn_visualizer
    attn_visualizer.clear()
    gr.Info("Attention visualization disabled")
    return "⏸️ Disabled"


def enable_gram_visualization(layer_indices_str="10,20,30,40,50"):
    """Enable gram matrix visualization by registering hooks"""
    global gram_visualizer, model
    if model is not None:
        try:
            # Parse layer indices
            layer_indices = [int(x.strip()) for x in layer_indices_str.split(',')] if layer_indices_str else None
            gram_visualizer.register_hooks(model.model.diffusion_model, layer_indices=layer_indices, clear_existing_data=False)
            gram_visualizer.reset_step()
            gr.Info(f"Gram matrix visualization enabled for layers: {layer_indices}")
            return f"✅ Enabled ({len(gram_visualizer.get_layer_names())} layers)"
        except Exception as e:
            gr.Error(f"Failed to enable: {str(e)}")
            return f"❌ Error: {str(e)}"
    else:
        gr.Warning("Please load a model first")
        return "❌ No model loaded"


def disable_gram_visualization():
    """Disable gram matrix visualization"""
    global gram_visualizer
    gram_visualizer.clear()
    gr.Info("Gram matrix visualization disabled")
    return "⏸️ Disabled"


def visualize_gram_matrices(
    reference_img,
    sketch_img,
    patch_row,
    patch_col,
    alpha,
    height=1024,
    width=1024,
    show_layer_label=True,
    denoising_step=-1
):
    """
    Visualize gram matrices for transformer block outputs
    Args:
        reference_img: Not used (kept for API compatibility)
        sketch_img: Sketch image - used as base for similarity overlay
        patch_row: Target patch row index
        patch_col: Target patch column index
        alpha: Overlay transparency (0.0-1.0)
        height: Image height
        width: Image width
        show_layer_label: Whether to display layer index on the visualization
        denoising_step: Denoising step to visualize (-1 = last step, 0 = first step, etc.)
    Returns:
        Tuple of (marked_sketch, gram_gallery):
        - marked_sketch: Sketch with target patch marked
        - gram_gallery: Gallery of gram similarity overlays on sketch for all layers
    Note:
        Gram matrix is computed from sketch features, so similarity is overlaid on sketch
    """
    global gram_visualizer
    
    try:
        from PIL import Image
        import numpy as np
        
        if not exists(sketch_img):
            gr.Warning("Please provide a sketch image")
            return None, []
        
        # Check if we have any captured gram matrices
        layer_names = gram_visualizer.get_layer_names()
        
        if not layer_names:
            gr.Warning("No gram matrices captured. Please run inference first with gram visualization enabled.")
            return None, []
        
        # Check available steps
        available_steps = gram_visualizer.get_available_steps()
        if not available_steps:
            gr.Warning("No gram matrices captured. Please run inference first with gram visualization enabled.")
            return None, []
        
        # Validate step
        if denoising_step != -1 and denoising_step not in available_steps:
            gr.Warning(f"Step {denoising_step} not available. Available steps: {available_steps}. Using last step.")
            denoising_step = -1
        
        print(f"Available denoising steps: {available_steps}, visualizing step: {denoising_step if denoising_step != -1 else 'last'}")
        
        # Calculate spatial size (latent space is 1/8 of image size)
        latent_h = height // 8
        latent_w = width // 8
        spatial_size = (latent_h, latent_w)
        
        # Ensure patch indices are within bounds
        patch_row = min(max(0, int(patch_row)), latent_h - 1)
        patch_col = min(max(0, int(patch_col)), latent_w - 1)
        target_patch = (patch_row, patch_col)
        
        print(f"\n=== Gram Matrix Visualization ===")
        print(f"Target patch: ({patch_row}, {patch_col})")
        print(f"Spatial size: {spatial_size}")
        print(f"Available layers: {layer_names}")
        
        # Prepare sketch image as base (gram matrix is computed from sketch features)
        if isinstance(sketch_img, str):
            sketch_np = np.array(Image.open(sketch_img).convert('RGB'))
        else:
            sketch_np = np.array(sketch_img)
        
        # Use original image dimensions for visualization
        orig_img_h, orig_img_w = sketch_np.shape[:2]
        
        # Mark target patch on sketch (in image coordinate space)
        marked_sketch = sketch_np.copy()
        
        # Note: The gram matrix might have different spatial resolution
        # We mark the patch in original image coordinate space based on latent space mapping
        # latent_h x latent_w patches map to the original image dimensions
        patch_h_size = orig_img_h / latent_h
        patch_w_size = orig_img_w / latent_w
        center_y = int((patch_row + 0.5) * patch_h_size)
        center_x = int((patch_col + 0.5) * patch_w_size)
        
        # Draw marker
        import cv2
        cv2.circle(marked_sketch, (center_x, center_y), radius=10, color=(255, 0, 0), thickness=-1)
        cv2.circle(marked_sketch, (center_x, center_y), radius=13, color=(255, 255, 255), thickness=3)
        
        # Draw patch boundary
        patch_top = int(patch_row * patch_h_size)
        patch_left = int(patch_col * patch_w_size)
        patch_bottom = int((patch_row + 1) * patch_h_size)
        patch_right = int((patch_col + 1) * patch_w_size)
        cv2.rectangle(marked_sketch, (patch_left, patch_top), (patch_right, patch_bottom), 
                     (0, 255, 0), thickness=2)
        
        print(f"Marked patch on sketch at image coords: center=({center_x}, {center_y})")
        
        marked_sketch_pil = Image.fromarray(marked_sketch)
        
        # Generate gram similarity overlays for all layers (or selected layer)
        results = []
        
        for idx, layer_name in enumerate(layer_names):
            gram = gram_visualizer.get_gram_matrix(layer_name, step=denoising_step)
            if gram is None:
                continue
            
            # Visualize similarity overlay on sketch (gram matrix is from sketch features)
            overlay = gram_visualizer.visualize_gram_similarity(
                gram_matrix=gram,
                image=sketch_np,
                target_patch_idx=target_patch,
                spatial_size=spatial_size,
                alpha=alpha
            )
            
            overlay_pil = Image.fromarray(overlay)
            
            # Add layer label if requested
            if show_layer_label:
                from PIL import ImageDraw, ImageFont
                draw = ImageDraw.Draw(overlay_pil)
                try:
                    font = ImageFont.truetype("arial.ttf", 56)
                except:
                    font = ImageFont.load_default()
                
                # Use the actual layer index from the full layer list
                actual_layer_idx = layer_names.index(layer_name)
                label_text = f"Layer {actual_layer_idx}: {layer_name}"
                draw.text((20, 20), label_text, fill=(255, 255, 0), font=font, stroke_width=4, stroke_fill=(0, 0, 0))
            
            results.append(overlay_pil)
        
        print(f"Generated {len(results)} gram similarity visualizations")
        gr.Info(f"Gram matrix visualization complete for {len(results)} layers")
        
        return marked_sketch_pil, results
        
    except Exception as e:
        import traceback
        print(f"Error in gram matrix visualization: {e}")
        traceback.print_exc()
        gr.Error(f"Visualization failed: {str(e)}")
        return None, []


def get_available_layers(attn_type="cross"):
    """Get list of available attention layers"""
    global attn_visualizer
    return attn_visualizer.get_layer_names(attn_type)


def visualize_attention_maps(
    reference_img,
    sketch_img,
    token_row,
    token_col,
    attn_type="cross",
    aggregate="mean",
    height=1024,
    width=1024,
    show_layer_label=True
):
    """
    Visualize attention maps for selected token
    Args:
        reference_img: Reference image
        sketch_img: Sketch image - token grid is based on this
        token_row: Row index of token in latent grid (query token)
        token_col: Column index of token in latent grid (query token)
        attn_type: 'cross' or 'self'
        aggregate: How to aggregate heads ('mean', 'max', or 'all')
        height: Image height
        width: Image width
        show_layer_label: Whether to display layer index on the visualization
    
    Returns:
        Tuple of (sketch_grid_image, attention_maps_list):
        - sketch_grid_image: Sketch with token grid overlay
        - attention_maps_list: For self-attention: attention maps on sketch showing which parts the selected token attends to
                               For cross-attention: attention maps on reference showing which parts the selected token attends to
    """
    global attn_visualizer
    
    try:
        if not exists(sketch_img):
            gr.Warning("Please provide a sketch image")
            return None, []
        
        # Grid size is fixed at 16 (each token = 16x16 pixels)
        # Grid count: height//16 x width//16
        grid_size = 16
        num_grids_h = height // grid_size
        num_grids_w = width // grid_size
        
        # Calculate token index from grid position (query token in sketch)
        token_idx = int(token_row * num_grids_w + token_col)
        
        print(f"Grid size: {grid_size}×{grid_size} pixels per token")
        print(f"Token count: {num_grids_h}×{num_grids_w} = {num_grids_h * num_grids_w} tokens")
        print(f"Selected token: ({int(token_row)}, {int(token_col)}) = index {token_idx}")
        
        # Create sketch grid with selected token highlighted
        sketch_grid_img = create_sketch_grid(sketch_img, height, width)
        
        # Check if we have any captured attention maps
        is_cross = (attn_type == "cross")
        attn_dict = attn_visualizer.cross_attn_maps if is_cross else attn_visualizer.self_attn_maps
        
        if not attn_dict:
            gr.Warning("No attention maps captured. Please run inference first with visualization enabled.")
            return sketch_grid_img, []
        
        # Get layer names
        layer_names = attn_visualizer.get_layer_names('cross' if is_cross else 'self')
        
        # Get sketch dimensions - all visualization results should match sketch size
        sketch_array = np.array(sketch_img)
        target_h, target_w = sketch_array.shape[:2]
        
        # Prepare base image for visualization (resize to match sketch dimensions)
        if is_cross:
            # Cross-attention: show attention on reference (KV side)
            if not exists(reference_img):
                gr.Warning("Please provide a reference image for cross-attention visualization")
                return sketch_grid_img, []
            ref_array = np.array(reference_img)
            # Resize reference to match sketch dimensions
            if ref_array.shape[:2] != (target_h, target_w):
                import cv2
                base_array = cv2.resize(ref_array, (target_w, target_h))
            else:
                base_array = ref_array
        else:
            # Self-attention: show attention on sketch (both Q and KV from sketch)
            base_array = sketch_array
        
        # Visualize
        results = []
        
        for layer_idx, layer_name in enumerate(layer_names[:8]):  # Limit to 8 layers for display
            attn_map = attn_visualizer.get_attention_map(layer_name, token_idx, is_cross)
            if attn_map is None:
                continue
            
            # Aggregate across heads if requested
            if aggregate == 'mean':
                attn_map = attn_map.mean(dim=0)
            elif aggregate == 'max':
                attn_map = attn_map.max(dim=0)[0]
            
            # Infer spatial size from attention map (KV side)
            n_tokens = attn_map.shape[-1] if len(attn_map.shape) > 1 else attn_map.shape[0]
            sqrt_n = np.sqrt(n_tokens)
            
            # Check if it's a perfect square
            if int(sqrt_n) ** 2 == n_tokens:
                spatial_h = int(sqrt_n)
                spatial_w = int(sqrt_n)
            else:
                # Factor the number into height and width
                found = False
                for h in range(int(sqrt_n), 0, -1):
                    if n_tokens % h == 0:
                        w = n_tokens // h
                        if not found or abs(h - w) < abs(spatial_h - spatial_w):
                            spatial_h = h
                            spatial_w = w
                            found = True
                            if h == w:
                                break
            
            # Debug: print token information
            if is_cross:
                print(f"Cross-attention KV tokens (reference): {n_tokens} = {spatial_h}×{spatial_w}")
            else:
                print(f"Self-attention tokens (sketch): {n_tokens} = {spatial_h}×{spatial_w}")
            
            # For self-attention, pass the query token index to draw red box on the same image
            # For cross-attention, we'll add text annotation instead
            query_token_for_box = token_idx if not is_cross else None
            
            # Visualize on appropriate image
            vis_images = attn_visualizer.visualize_attention_on_image(
                attn_map, base_array, 
                spatial_size=(spatial_h, spatial_w),
                source_token_idx=query_token_for_box
            )
            
            # Add to results
            for img in vis_images:
                img_pil = Image.fromarray(img.astype(np.uint8))
                
                # Add layer label if requested
                if show_layer_label:
                    from PIL import ImageDraw, ImageFont
                    draw = ImageDraw.Draw(img_pil)
                    try:
                        font = ImageFont.truetype("arial.ttf", 56)
                    except:
                        font = ImageFont.load_default()
                    
                    label_text = f"Layer {layer_idx}: {layer_name}"
                    draw.text((20, 20), label_text, fill=(255, 255, 0), font=font, stroke_width=4, stroke_fill=(0, 0, 0))
                
                results.append(img_pil)
        
        if len(results) == 0:
            gr.Warning("No attention maps to visualize")
            return sketch_grid_img, []
        
        return sketch_grid_img, results
        
    except Exception as e:
        gr.Warning(f"Error: {str(e)}")
        print(f"Attention visualization error: {e}")
        import traceback
        traceback.print_exc()
        return None, []


def create_sketch_grid(sketch_img, height=1024, width=1024):
    """Create a grid overlay on sketch image to show token positions"""
    if not exists(sketch_img):
        return None
    
    # Grid size is fixed at 16 (each token = 16x16 pixels)
    grid_size = 16
    num_grids_h = height // grid_size
    num_grids_w = width // grid_size
    
    img_array = np.array(sketch_img).copy()
    img_h, img_w = img_array.shape[:2]
    
    # Calculate cell size in image space
    cell_h = img_h // num_grids_h
    cell_w = img_w // num_grids_w
    
    # Draw grid lines
    for i in range(num_grids_h + 1):
        # Horizontal lines
        y = i * cell_h
        cv2.line(img_array, (0, y), (img_w, y), (0, 255, 0), 2)
    for i in range(num_grids_w + 1):
        # Vertical lines
        x = i * cell_w
        cv2.line(img_array, (x, 0), (x, img_h), (0, 255, 0), 2)
    
    gr.Info(f"Grid: {grid_size}×{grid_size} pixels/token, Token count: {num_grids_h}×{num_grids_w} = {num_grids_h * num_grids_w}")
    return Image.fromarray(img_array)


def select_token_by_click(sketch_img, height, width, evt: gr.SelectData):
    """Convert mouse click coordinates to token position"""
    if not exists(sketch_img):
        return gr.update(), gr.update()
    
    # Get click coordinates
    click_x, click_y = evt.index
    
    # Get image dimensions
    img_array = np.array(sketch_img)
    img_h, img_w = img_array.shape[:2]
    
    # Calculate token grid
    grid_size = 16
    num_grids_h = height // grid_size
    num_grids_w = width // grid_size
    
    # Calculate which token was clicked
    token_col = int((click_x / img_w) * num_grids_w)
    token_row = int((click_y / img_h) * num_grids_h)
    
    # Clamp to valid range
    token_row = max(0, min(token_row, num_grids_h - 1))
    token_col = max(0, min(token_col, num_grids_w - 1))
    
    print(f"Click at ({click_x}, {click_y}) -> Token ({token_row}, {token_col})")
    gr.Info(f"Selected token: ({token_row}, {token_col})")
    
    return gr.update(value=token_row), gr.update(value=token_col)


def highlight_selected_token(sketch_img, token_row, token_col, num_grids_w, num_grids_h):
    """Highlight the selected token on sketch image"""
    if not exists(sketch_img):
        return None
    
    img_array = np.array(sketch_img).copy()
    img_h, img_w = img_array.shape[:2]
    
    # Calculate cell size in image space
    cell_h = img_h // num_grids_h
    cell_w = img_w // num_grids_w
    
    # Calculate token position
    y_start = int(token_row * cell_h)
    y_end = int((token_row + 1) * cell_h)
    x_start = int(token_col * cell_w)
    x_end = int((token_col + 1) * cell_w)
    
    # Draw grid
    for i in range(num_grids_h + 1):
        y = i * cell_h
        cv2.line(img_array, (0, y), (img_w, y), (200, 200, 200), 1)
    for i in range(num_grids_w + 1):
        x = i * cell_w
        cv2.line(img_array, (x, 0), (x, img_h), (200, 200, 200), 1)
    
    # Highlight selected token with thick border
    cv2.rectangle(img_array, (x_start, y_start), (x_end, y_end), (255, 0, 0), 4)
    
    # Add semi-transparent overlay
    overlay = img_array.copy()
    cv2.rectangle(overlay, (x_start, y_start), (x_end, y_end), (255, 255, 0), -1)
    cv2.addWeighted(overlay, 0.3, img_array, 0.7, 0, img_array)
    
    return Image.fromarray(img_array)


def select_patch_by_click(sketch_img, height, width, evt: gr.SelectData):
    """Convert mouse click coordinates to patch position for gram matrix visualization"""
    if not exists(sketch_img):
        return gr.update(), gr.update()
    
    # Get click coordinates
    click_x, click_y = evt.index
    
    # Get image dimensions
    img_array = np.array(sketch_img)
    img_h, img_w = img_array.shape[:2]
    
    # Calculate patch grid (latent space is 1/8 of image size)
    latent_h = height // 8
    latent_w = width // 8
    
    # Calculate which patch was clicked
    patch_col = int((click_x / img_w) * latent_w)
    patch_row = int((click_y / img_h) * latent_h)
    
    # Clamp to valid range
    patch_row = max(0, min(patch_row, latent_h - 1))
    patch_col = max(0, min(patch_col, latent_w - 1))
    
    print(f"Click at ({click_x}, {click_y}) -> Patch ({patch_row}, {patch_col})")
    gr.Info(f"Selected patch: ({patch_row}, {patch_col})")
    
    return gr.update(value=patch_row), gr.update(value=patch_col)


def create_patch_grid(sketch_img, height=1024, width=1024):
    """Create a grid overlay on sketch for patch selection"""
    if not exists(sketch_img):
        return None
    
    # Convert to numpy array
    if isinstance(sketch_img, str):
        from PIL import Image
        img = Image.open(sketch_img).convert('RGB')
        img_array = np.array(img)
    else:
        img_array = np.array(sketch_img)
    
    img_h, img_w = img_array.shape[:2]
    
    # Calculate patch grid dimensions
    latent_h = height // 8
    latent_w = width // 8
    
    # Calculate cell size in image space
    cell_h = img_h / latent_h
    cell_w = img_w / latent_w
    
    # Draw grid lines
    img_with_grid = img_array.copy()
    
    # Draw horizontal lines
    for i in range(latent_h + 1):
        y = int(i * cell_h)
        cv2.line(img_with_grid, (0, y), (img_w, y), (150, 150, 150), 1)
    
    # Draw vertical lines
    for i in range(latent_w + 1):
        x = int(i * cell_w)
        cv2.line(img_with_grid, (x, 0), (x, img_h), (150, 150, 150), 1)
    
    # Add text overlay
    from PIL import Image, ImageDraw, ImageFont
    pil_img = Image.fromarray(img_with_grid)
    draw = ImageDraw.Draw(pil_img)
    
    try:
        # Try to use a nice font
        font = ImageFont.truetype("arial.ttf", 48)
    except:
        font = ImageFont.load_default()
    
    text = f"Patch Grid: {latent_h}×{latent_w} (Click to select)"
    draw.text((20, 20), text, fill=(255, 255, 255), font=font, stroke_width=4, stroke_fill=(0, 0, 0))
    
    print(f"Created patch grid: {latent_h}×{latent_w} patches")
    
    return pil_img


def set_cas_scales(accurate, cas_args):
    enc_scale, middle_scale, low_scale, strength = cas_args[:4]
    attn_scales = cas_args[5:]
    if not accurate:
        scale_strength = {
            "level_control": True,
            "scales": {
                "encoder": enc_scale * strength,
                "middle": middle_scale * strength,
                "low": low_scale * strength,
            }
        }
    else:
        scale_strength = {
            "level_control": False,
            "scales": list(attn_scales)
        }
    return scale_strength


@torch.no_grad()
def inference(
        style_enhance, bg_enhance, fg_enhance, fg_disentangle_scale,
        bs, input_s, input_r, input_bg, mask_ts, mask_ss, gs_r, gs_s, ctl_scale,
        ctl_scale_1, ctl_scale_2, ctl_scale_3, ctl_scale_4,
        fg_strength, bg_strength, merge_scale, mask_scale, height, width, seed, low_vram, step,
        injection, autofit_size, remove_fg, rmbg, latent_inpaint, infid_x, infid_r, injstep, crop, pad_scale,
        start_step, end_step, no_start_step, no_end_step, return_inter, sampler, scheduler, preprocess,
        deterministic, text, target, anchor, control, target_scale, ts0, ts1, ts2, ts3, enhance, accurate,
        *args
):
    global global_seed, line_extractor, mask_extractor, gram_visualizer
    global_seed = seed if seed > -1 else random.randint(0, MAXM_INT32)
    torch.manual_seed(global_seed)
    
    # Auto-fit size based on sketch dimensions
    if autofit_size and exists(input_s):
        sketch_w, sketch_h = input_s.size
        aspect_ratio = sketch_w / sketch_h
        
        # Target area (default 1024x1024 = 1048576)
        target_area = 1024 * 1024
        
        # Calculate dimensions that maintain aspect ratio
        new_h = int((target_area / aspect_ratio) ** 0.5)
        new_w = int(new_h * aspect_ratio)
        
        # Round to nearest multiple of 32
        height = ((new_h + 16) // 32) * 32
        width = ((new_w + 16) // 32) * 32
        
        # Clamp to valid range [768, 1536]
        height = max(768, min(1536, height))
        width = max(768, min(1536, width))
        
        gr.Info(f"Auto-fitted size: {width}x{height} (original: {sketch_w}x{sketch_h})")
    
    # Reset gram visualizer step counter before inference and attach to model
    if gram_visualizer.enabled:
        gram_visualizer.reset_step()
        model.gram_visualizer = gram_visualizer

    # if vis_crossattn:
    #     assert (height, width) == (512, 512), "Only to visualize standard results at 16 latent scale"
    #     visualizer.hack(model, vh, vw, width)
    smask, rmask, bgmask = None, None, None
    manipulation_params = parse_prompts(text, target, anchor, control, target_scale, ts0, ts1, ts2, ts3, enhance)
    inputs = preprocessing_inputs(
        sketch = input_s,
        reference = input_r,
        background = input_bg,
        preprocess = preprocess,
        hook = injection,
        resolution = (height, width),
        extractor = line_extractor,
        pad_scale = pad_scale,
    )
    sketch, reference, background, original_shape, inject_xr, inject_xs, white_sketch = inputs
    if not osp.exists(attn_vispath):
        os.makedirs(attn_vispath)

    cond = {"reference": reference, "sketch": sketch, "background": background}
    mask_guided = bg_enhance or fg_enhance

    if exists(white_sketch) and exists(reference) and mask_guided:
        mask_extractor.cuda()
        smask_extractor.cuda()
        smask = smask_extractor.proceed(
            x=white_sketch, pil_x=input_s, th=height, tw=width, threshold=mask_ss, crop=False
        )

        if exists(background) and remove_fg:
            bgmask = mask_extractor.proceed(x=background, pil_x=input_bg, threshold=mask_ts, dilate=True)
            filtered_background = torch.where(bgmask < mask_ts, background, torch.ones_like(background))
            cond.update({"background": filtered_background, "rmask": bgmask})
            
        else:
            rmask = mask_extractor.proceed(x=reference, pil_x=input_r, threshold=mask_ts, dilate=True)
            cond.update({"rmask": rmask})
        rmask = torch.where(rmask > 0.5, torch.ones_like(rmask), torch.zeros_like(rmask))
        # cond.update({"smask": scaled_resize(smask, 0.125)})
        cond.update({"smask": smask})
        smask_extractor.cpu()
        mask_extractor.cpu()

    # if hasattr(model.cond_stage_model, "scale_factor") and scale_factor != model.cond_stage_model.scale_factor:
    #     model.cond_stage_model.update_scale_factor(scale_factor)
    scale_strength = set_cas_scales(accurate, args)
    ctl_scales = [ctl_scale_1, ctl_scale_2, ctl_scale_3, ctl_scale_4]
    ctl_scales = [t * ctl_scale for t in ctl_scales]

    results = model.generate(
        # Colorization mode
        style_enhance = style_enhance,
        bg_enhance = bg_enhance,
        fg_enhance = fg_enhance,
        fg_disentangle_scale = fg_disentangle_scale,
        latent_inpaint = latent_inpaint,

        # Conditional inputs
        cond = cond,
        ctl_scale = ctl_scales,
        merge_scale = merge_scale,
        mask_scale = mask_scale,
        mask_thresh = mask_ts,
        mask_thresh_sketch = mask_ss,

        # Sampling settings
        bs = bs,
        gs = [gs_r, gs_s],
        sampler = sampler,
        scheduler = scheduler,
        start_step = start_step,
        end_step = end_step,
        no_start_step = no_start_step,
        no_end_step = no_end_step,
        strength = scale_strength,
        fg_strength = fg_strength,
        bg_strength = bg_strength,
        seed = global_seed,
        deterministic = deterministic,
        height = height,
        width = width,
        step = step,

        # Injection settings
        injection = injection,
        injection_cfg = infid_r,
        injection_control = infid_x,
        injection_start_step = injstep,
        hook_xr = inject_xr,
        hook_xs = inject_xs,

        # Additional settings
        low_vram = low_vram,
        return_intermediate = return_inter,
        manipulation_params = manipulation_params,
    )

    if rmbg:
        mask_extractor.cuda()
        mask = smask_extractor.proceed(x=-sketch, threshold=mask_ss).repeat(results.shape[0], 1, 1, 1)
        results = torch.where(mask >= mask_ss, results, torch.ones_like(results))
        mask_extractor.cpu()

    results = postprocess(results, sketch, reference, background, crop, original_shape,
                          mask_guided, smask, rmask, bgmask, mask_ts, mask_ss)
    torch.cuda.empty_cache()
    gr.Info("Generation completed.")
    return results
