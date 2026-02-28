"""Gradio web UI for ColorizeDiffusion XL.
Launch with: python -u app.py
Options:
    -manipulate    Enable text manipulation controls
    --full         Show advanced inference controls (per-level sketch strengths, cross-attention scales, etc.)
    --share        Create a public Gradio share link
"""

import gradio as gr
import argparse

from refnet.sampling import get_noise_schedulers, get_sampler_list
from functools import partial
from backend import *

links = {
    "base": "https://arxiv.org/abs/2401.01456",
    "v1": "https://openaccess.thecvf.com/content/WACV2025/html/Yan_ColorizeDiffusion_Improving_Reference-Based_Sketch_Colorization_with_Latent_Diffusion_Model_WACV_2025_paper.html",
    "v1.5": "https://arxiv.org/abs/2502.19937v1",
    "v2": "https://arxiv.org/abs/2504.06895",
    "xl": "https://arxiv.org/abs/2601.04883",
    "weights": "https://huggingface.co/tellurion/colorizer/tree/main",
    "github": "https://github.com/tellurion-kanata/colorizeDiffusion",
}

def app_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_name", '-addr', type=str, default="0.0.0.0")
    parser.add_argument("--server_port", '-port', type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--not_show_error", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--enable_text_manipulation", '-manipulate', action="store_true")
    parser.add_argument("--collect_feedback", '-cf', action="store_true")
    parser.add_argument("--full", action="store_true")
    return parser.parse_args()


def init_interface(opt, *args, **kwargs) -> None:
    sampler_list = get_sampler_list()
    scheduler_list = get_noise_schedulers()

    img_block = partial(gr.Image, type="pil", height=300, interactive=True, show_label=True, format="png")
    with gr.Blocks(
        title = "Colorize Diffusion",
        css_paths = "backend/style.css",
        theme = gr.themes.Ocean(),
        elem_id = "main-interface",
        analytics_enabled = False,
        fill_width = True
    ) as block:
        with gr.Row(elem_id="header-row", equal_height=True, variant="panel"):
            gr.Markdown(f"""<div class="header-container">
                <div class="app-header"><span class="emoji">🎨</span><span class="title-text">Colorize Diffusion</span></div>
                <div class="paper-links-icons">
                    <a href="{links['base']}" target="_blank">
                        <img src="https://img.shields.io/badge/arXiv-2407.15886 (base)-B31B1B?style=flat&logo=arXiv" alt="arXiv Paper">
                    </a>
                    <a href="{links['v1']}" target="_blank">
                        <img src="https://img.shields.io/badge/WACV 2025-v1-0CA4A5?style=flat&logo=Semantic%20Web" alt="WACV 2025">
                    </a>
                    <a href="{links['v1.5']}" target="_blank">
                        <img src="https://img.shields.io/badge/CVPR 2025-v1.5-0CA4A5?style=flat&logo=Semantic%20Web" alt="CVPR 2025">
                    </a>
                    <a href="{links['v2']}" target="_blank">
                        <img src="https://img.shields.io/badge/arXiv-2504.06895 (v2)-B31B1B?style=flat&logo=arXiv" alt="arXiv v2 Paper">
                    </a>
                    <a href="{links['weights']}" target="_blank">
                        <img src="https://img.shields.io/badge/Hugging%20Face-Model%20Weights-FF9D00?style=flat&logo=Hugging%20Face" alt="Model Weights">
                    </a>
                    <a href="{links['github']}" target="_blank">
                        <img src="https://img.shields.io/badge/GitHub-Repo-blue?style=flat&logo=GitHub" alt="GitHub">
                    </a>
                    <a href="https://github.com/tellurion-kanata/colorizeDiffusion/blob/master/LICENSE" target="_blank">
                        <img src="https://img.shields.io/badge/License-CC--BY--NC--SA%204.0-4CAF50?style=flat&logo=Creative%20Commons" alt="License">
                    </a>
                </div>
            </div>""")

        with gr.Row(elem_id="content-row", equal_height=False, variant="panel"):
            with gr.Column():
                with gr.Row(visible=opt.enable_text_manipulation):
                    target = gr.Textbox(label="Target prompt", value="", scale=2)
                    anchor = gr.Textbox(label="Anchor prompt", value="", scale=2)
                    control = gr.Textbox(label="Control prompt", value="", scale=2)
                with gr.Row(visible=opt.enable_text_manipulation):
                    target_scale = gr.Slider(label="Target scale", value=0.0, minimum=0, maximum=15.0, step=0.25, scale=2)
                    ts0 = gr.Slider(label="Threshold 0", value=0.5, minimum=0, maximum=1.0, step=0.01)
                    ts1 = gr.Slider(label="Threshold 1", value=0.55, minimum=0, maximum=1.0, step=0.01)
                    ts2 = gr.Slider(label="Threshold 2", value=0.65, minimum=0, maximum=1.0, step=0.01)
                    ts3 = gr.Slider(label="Threshold 3", value=0.95, minimum=0, maximum=1.0, step=0.01)
                with gr.Row(visible=opt.enable_text_manipulation):
                    enhance = gr.Checkbox(label="Enhance manipulation", value=False)
                    add_prompt = gr.Button(value="Add")
                    clear_prompt = gr.Button(value="Clear")
                    vis_button = gr.Button(value="Visualize")
                text_prompt = gr.Textbox(label="Final prompt", value="", lines=3, visible=opt.enable_text_manipulation)

                with gr.Row():
                    sketch_img = img_block(label="Sketch")
                    reference_img = img_block(label="Reference")
                    background_img = img_block(label="Background")

                style_enhance = gr.State(False)
                fg_enhance = gr.State(False)
                with gr.Row():
                    bg_enhance = gr.Checkbox(label="Low-level injection", value=False)
                    injection = gr.Checkbox(label="Attention injection", value=False)
                    autofit_size = gr.Checkbox(label="Autofit size", value=False)
                with gr.Row():
                    gs_r = gr.Slider(label="Reference guidance scale", minimum=1, maximum=15.0, value=4.0, step=0.5)
                    strength = gr.Slider(label="Reference strength", minimum=0, maximum=1, value=1, step=0.05)
                    fg_strength = gr.Slider(label="Foreground strength", minimum=0, maximum=1, value=1, step=0.05)
                    bg_strength = gr.Slider(label="Background strength", minimum=0, maximum=1, value=1, step=0.05)
                with gr.Row():
                    gs_s = gr.Slider(label="Sketch guidance scale", minimum=1, maximum=5.0, value=1.0, step=0.1)
                    ctl_scale = gr.Slider(label="Sketch strength", minimum=0, maximum=3, value=1, step=0.05)
                    mask_scale = gr.Slider(label="Background factor", minimum=0, maximum=2, value=1, step=0.05)
                    merge_scale = gr.Slider(label="Merging scale", minimum=0, maximum=1, value=0, step=0.05)
                with gr.Row():
                    bs = gr.Slider(label="Batch size", minimum=1, maximum=8, value=1, step=1, scale=1)
                    width = gr.Slider(label="Width", minimum=768, maximum=1536, value=1024, step=32, scale=2)
                with gr.Row():
                    step = gr.Slider(label="Step", minimum=1, maximum=100, value=20, step=1, scale=1)
                    height = gr.Slider(label="Height", minimum=768, maximum=1536, value=1024, step=32, scale=2)
                with gr.Row(visible=opt.full):
                    injection_control_scale = gr.Slider(label="Injection fidelity (sketch)", minimum=0.0, maximum=2.0,
                                                        value=0, step=0.05)
                    injection_fidelity = gr.Slider(label="Injection fidelity (reference) ", minimum=0.0, maximum=1.0,
                                                   value=0.5, step=0.05)
                    injection_start_step = gr.Slider(label="Injection start step ", minimum=0.0, maximum=1.0,
                                                     value=0, step=0.05)
                with gr.Row(visible=opt.full):
                    ctl_scale_1 = gr.Slider(label="Sketch strength 1", minimum=0, maximum=3, value=1, step=0.05)
                    ctl_scale_2 = gr.Slider(label="Sketch strength 2", minimum=0, maximum=3, value=1, step=0.05)
                    ctl_scale_3 = gr.Slider(label="Sketch strength 3", minimum=0, maximum=3, value=1, step=0.05)
                    ctl_scale_4 = gr.Slider(label="Sketch strength 4", minimum=0, maximum=3, value=1, step=0.05)
                with gr.Row(visible=opt.full):
                    low_scale = gr.Slider(label="Semantics crossattn scale",
                                          minimum=0, maximum=1, step=0.05, value=1)
                    middle_scale = gr.Slider(label="Color crossattn scale", minimum=0,
                                             maximum=1, step=0.05, value=1)
                    enc_scale = gr.Slider(label="Encoder crossattn scale",
                                           minimum=0, maximum=1, step=0.05, value=1)
                    fg_disentangle_scale = gr.Slider(label="Disentangle scale",
                                                     minimum=0, maximum=1, step=0.05, value=1)

                seed = gr.Slider(label="Seed", minimum=-1, maximum=MAXM_INT32, step=1, value=-1)
                with gr.Accordion("🔧 Advanced Settings", open=True):
                    with gr.Row():
                        crop = gr.Checkbox(label="Crop result", value=False, scale=1)
                        remove_fg = gr.Checkbox(label="Remove foreground in background input", value=False, scale=2)
                        rmbg = gr.Checkbox(label="Remove background in result", value=False, scale=2)
                        latent_inpaint = gr.Checkbox(
                            label = "Latent copy BG input",
                            value = False,
                            scale = 2
                        )
                    with gr.Row(visible=opt.full):
                        start_step = gr.Slider(label="Guidance start step", minimum=0, maximum=1, value=0,
                                               step=0.05)
                        end_step = gr.Slider(label="Guidance end step", minimum=0, maximum=1, value=1,
                                             step=0.05)
                        no_start_step = gr.Slider(label="No guidance start step", minimum=-0.05, maximum=1, value=-0.05,
                                                  step=0.05)
                        no_end_step = gr.Slider(label="No guidance end step", minimum=-0.05, maximum=1, value=-0.05,
                                                step=0.05)
                with gr.Row():
                    reuse_seed = gr.Button(value="♻️ Reuse Seed")
                    random_seed = gr.Button(value="🎲 Random Seed")
                    update_ckpts = gr.Button(value="🔄 Refresh Models")
                with gr.Accordion(
                        "Accurate control on crossattn scale (only for SD2.1)",
                        open=True,
                        visible=opt.full
                ):
                    accurate = gr.Checkbox(label="Activate accurate crossattn control", value=False)
                    with gr.Row():
                        attn0 = gr.Slider(label="0.down0.attn0", minimum=0, maximum=1, step=0.05, value=1)
                        attn1 = gr.Slider(label="1.down0.attn1", minimum=0, maximum=1, step=0.05, value=1)
                        attn2 = gr.Slider(label="2.down1.attn0", minimum=0, maximum=1, step=0.05, value=1)
                        attn3 = gr.Slider(label="3.down1.attn1", minimum=0, maximum=1, step=0.05, value=1)
                    with gr.Row():
                        attn4 = gr.Slider(label="4.down2.attn0", minimum=0, maximum=1, step=0.05, value=1)
                        attn5 = gr.Slider(label="5.down2.attn1", minimum=0, maximum=1, step=0.05, value=1)
                        attn6 = gr.Slider(label="6.middle", minimum=0, maximum=1, step=0.05, value=1)
                        attn7 = gr.Slider(label="7.up2.attn0", minimum=0, maximum=1, step=0.05, value=1)
                    with gr.Row():
                        attn8 = gr.Slider(label="8.up2.attn1", minimum=0, maximum=1, step=0.05, value=1)
                        attn9 = gr.Slider(label="9.up2.attn2", minimum=0, maximum=1, step=0.05, value=1)
                        attn10 = gr.Slider(label="10.up1.attn0", minimum=0, maximum=1, step=0.05, value=1)
                        attn11 = gr.Slider(label="11.up1.attn1", minimum=0, maximum=1, step=0.05, value=1)
                    with gr.Row():
                        attn12 = gr.Slider(label="12.up1.attn2", minimum=0, maximum=1, step=0.05, value=1)
                        attn13 = gr.Slider(label="13.up0.attn0", minimum=0, maximum=1, step=0.05, value=1)
                        attn14 = gr.Slider(label="14.up0.attn1", minimum=0, maximum=1, step=0.05, value=1)
                        attn15 = gr.Slider(label="15.up0.attn2", minimum=0, maximum=1, step=0.05, value=1)

            with gr.Column():
                result_gallery = gr.Gallery(
                    label='Output', show_label=False, elem_id="gallery", preview=True, type="pil", format="png"
                )
                run_button = gr.Button("🚀 Generate", variant="primary", size="lg")
                with gr.Row():
                    mask_ts = gr.Slider(label="Reference mask threshold", minimum=0., maximum=1., value=0.5, step=0.01)
                    mask_ss = gr.Slider(label="Sketch mask threshold", minimum=0., maximum=1., value=0.05, step=0.01)
                    pad_scale = gr.Slider(label="Reference padding scale", minimum=1, maximum=2, value=1, step=0.05)

                with gr.Row():
                    sd_model = gr.Dropdown(choices=get_checkpoints(), label="Models", value=get_checkpoints()[0])
                    extractor_model = gr.Dropdown(choices=line_extractor_list,
                                              label="Line extractor", value=default_line_extractor)
                    mask_model = gr.Dropdown(choices=mask_extractor_list, label="Reference mask extractor",
                                             value=default_mask_extractor)
                with gr.Row():
                    sampler = gr.Dropdown(choices=sampler_list, value="DPM++ 3M SDE", label="Sampler")
                    scheduler = gr.Dropdown(choices=scheduler_list, value=scheduler_list[0], label="Noise scheduler")
                    preprocessor = gr.Dropdown(choices=["none", "extract", "invert", "invert-webui"],
                                               label="Sketch preprocessor", value="invert")
                                               
                with gr.Accordion("🔍 Attention Map Visualization", open=False):
                    visualize_attn_btn = gr.Button("🎨 Visualize Attention", variant="primary")
                    attn_vis_status = gr.Textbox(label="Status", value="⏸️ Disabled", interactive=False, scale=2)
                    with gr.Row():
                        enable_attn_vis = gr.Button("▶️ Enable Attention Capture", size="sm")
                        disable_attn_vis = gr.Button("⏸️ Disable Attention Capture", size="sm")
                        show_attn_layer_label = gr.Checkbox(label="Show layer index", value=True, scale=1)
                    with gr.Row():
                        token_row = gr.Slider(label="Query Row", minimum=0, maximum=95, value=32, step=1, scale=2)
                        token_col = gr.Slider(label="Query Column", minimum=0, maximum=95, value=32, step=1, scale=2)
                        attn_type = gr.Radio(
                            choices=["cross", "self"],
                            value="cross",
                            label="Attention Type",
                            scale=2
                        )
                        head_aggregate = gr.Radio(
                            choices=["mean", "max", "all"],
                            value="mean",
                            label="Head Aggregation",
                            scale=3
                        )
                    with gr.Row():
                        sketch_grid = gr.Image(label="Sketch with Token Grid (Click to Select)", type="pil", interactive=False)
                        attention_gallery = gr.Gallery(
                            label='Attention Maps',
                            show_label=True,
                            elem_id="attention_gallery",
                            preview=True,
                            type="pil",
                            format="png",
                            columns=6,
                            height=300
                        )
                
                with gr.Accordion("📊 Gram Matrix Visualization", open=False):
                    with gr.Row():
                        visualize_gram_btn = gr.Button("🔬 Visualize Gram Matrix", variant="primary", scale=2)
                        show_patch_grid_btn = gr.Button("🔲 Show Patch Grid", variant="secondary", scale=1)
                    gram_vis_status = gr.Textbox(label="Status", value="⏸️ Disabled", interactive=False)
                    gram_layer_indices = gr.Textbox(label="Layer Indices", value="20,21,22,23,24,60,61,62,63,64", scale=3)
                    with gr.Row():
                        enable_gram_vis = gr.Button("▶️ Enable Gram Capture", size="sm")
                        disable_gram_vis = gr.Button("⏸️ Disable Gram Capture", size="sm")
                        show_gram_layer_label = gr.Checkbox(label="Show layer index", value=True, scale=1)
                    with gr.Row():
                        gram_patch_row = gr.Slider(label="Target Patch Row", minimum=0, maximum=127, value=64, step=1, scale=2)
                        gram_patch_col = gr.Slider(label="Target Patch Col", minimum=0, maximum=127, value=64, step=1, scale=2)
                        gram_denoising_step = gr.Slider(label="Denoising Step", minimum=-1, maximum=50, value=-1, step=1, scale=2)
                        gram_alpha = gr.Slider(label="Overlay Alpha", minimum=0.0, maximum=1.0, value=0.6, step=0.05, scale=2)
                    with gr.Row():
                        gram_sketch_grid = gr.Image(label="Sketch with Patch Grid (Click to Select)", type="pil", interactive=False)
                        gram_sketch_marked = gr.Image(label="Sketch with Target Patch", type="pil", interactive=False)
                    gram_gallery = gr.Gallery(
                        label='Gram Matrix Similarity Maps',
                        show_label=True,
                        elem_id="gram_gallery",
                        preview=True,
                        type="pil",
                        format="png",
                        columns=6,
                        height=300
                    )
                
                with gr.Row(visible=opt.collect_feedback):
                    colorization_rating = gr.Slider(label="Colorization Effect", minimum=1, maximum=5, value=3, step=1)
                    sketch_fidelity_rating = gr.Slider(label="Sketch Fidelity", minimum=1, maximum=5, value=3, step=1)
                    reference_similarity_rating = gr.Slider(label="Reference Similarity", minimum=1, maximum=5, value=3, step=1)
                with gr.Row(visible=opt.collect_feedback):
                    save_button = gr.Button("💾 Save Results", variant="secondary", size="lg")
                    delete_button = gr.Button("🗑️ Delete Last", variant="secondary", size="lg")
                
                save_status = gr.Textbox(label="Save Status", value="", interactive=False, visible=opt.collect_feedback)

                with gr.Row():
                    vae_fp16 = gr.Button(value="fp16 vae")
                    vae_fp32 = gr.Button(value="fp32 vae")
                    fp16 = gr.Button(value="fp16 unet")
                    fp32 = gr.Button(value="fp32 unet")

                with gr.Row():
                    deterministic = gr.Checkbox(label="Deterministic batch seed", value=False)
                    save_memory = gr.Checkbox(label="Save memory", value=False)
                    return_inter = gr.Checkbox(label="Check intermediates (only for ddim)", value=False)

        add_prompt.click(fn=apppend_prompt,
                         inputs=[target, anchor, control, target_scale, enhance, ts0, ts1, ts2, ts3, text_prompt],
                         outputs=[target, anchor, control, target_scale, enhance, ts0, ts1, ts2, ts3, text_prompt])
        clear_prompt.click(fn=clear_prompts, outputs=[text_prompt])

        reuse_seed.click(fn=get_last_seed, outputs=[seed])
        random_seed.click(fn=reset_random_seed, outputs=[seed])
        update_ckpts.click(fn=update_models, outputs=[sd_model])

        extractor_model.input(fn=switch_extractor, inputs=[extractor_model])
        sd_model.input(fn=load_model, inputs=[sd_model])
        mask_model.input(fn=switch_mask_extractor, inputs=[mask_model])

        fp16.click(fn=switch_to_fp16)
        fp32.click(fn=switch_to_fp32)
        vae_fp16.click(fn=switch_vae_to_fp16)
        vae_fp32.click(fn=switch_vae_to_fp32)

        ips = [style_enhance, bg_enhance, fg_enhance, fg_disentangle_scale,
               bs, sketch_img, reference_img, background_img, mask_ts, mask_ss, gs_r, gs_s, ctl_scale,
               ctl_scale_1, ctl_scale_2, ctl_scale_3, ctl_scale_4, fg_strength, bg_strength, merge_scale,
               mask_scale, height, width, seed, save_memory, step, injection, autofit_size,
               remove_fg, rmbg, latent_inpaint, injection_control_scale, injection_fidelity, injection_start_step,
               crop, pad_scale, start_step, end_step, no_start_step, no_end_step, return_inter, sampler, scheduler,
               preprocessor, deterministic, text_prompt, target, anchor, control, target_scale, ts0, ts1, ts2, ts3,
               enhance, accurate, enc_scale, middle_scale, low_scale, strength, attn0, attn1, attn2, attn3,
               attn4, attn5, attn6, attn7, attn8, attn9, attn10, attn11, attn12, attn13, attn14, attn15]

        # Configure the inference function with proper queue settings
        run_button.click(
            fn = inference,
            inputs = ips,
            outputs = [result_gallery],
        )
        
        vis_button.click(
            fn = visualize,
            inputs = [reference_img, text_prompt, control, ts0, ts1, ts2, ts3],
            outputs = [result_gallery],
        )
        
        save_button.click(
            fn = save_results,
            inputs = [sketch_img, reference_img, result_gallery, colorization_rating, sketch_fidelity_rating, reference_similarity_rating],
            outputs = [save_status],
        )
        
        delete_button.click(
            fn = delete_last_result,
            inputs = [],
            outputs = [save_status],
        )
        
        enable_attn_vis.click(
            fn = enable_attention_visualization,
            inputs = [],
            outputs = [attn_vis_status],
        )
        
        disable_attn_vis.click(
            fn = disable_attention_visualization,
            inputs = [],
            outputs = [attn_vis_status],
        )
        
        sketch_grid.select(
            fn = select_token_by_click,
            inputs = [sketch_img, height, width],
            outputs = [token_row, token_col],
        )
        
        visualize_attn_btn.click(
            fn = visualize_attention_maps,
            inputs = [reference_img, sketch_img, token_row, token_col, attn_type, head_aggregate, height, width, show_attn_layer_label],
            outputs = [sketch_grid, attention_gallery],
        )
        
        enable_gram_vis.click(
            fn = enable_gram_visualization,
            inputs = [gram_layer_indices],
            outputs = [gram_vis_status],
        )
        
        disable_gram_vis.click(
            fn = disable_gram_visualization,
            inputs = [],
            outputs = [gram_vis_status],
        )
        
        show_patch_grid_btn.click(
            fn = create_patch_grid,
            inputs = [sketch_img, height, width],
            outputs = [gram_sketch_grid],
        )
        
        gram_sketch_grid.select(
            fn = select_patch_by_click,
            inputs = [sketch_img, height, width],
            outputs = [gram_patch_row, gram_patch_col],
        )
        
        visualize_gram_btn.click(
            fn = visualize_gram_matrices,
            inputs = [reference_img, sketch_img, gram_patch_row, gram_patch_col, gram_alpha, height, width, show_gram_layer_label, gram_denoising_step],
            outputs = [gram_sketch_marked, gram_gallery],
        )

        block.launch(
            server_name = opt.server_name,
            share = opt.share,
            server_port = opt.server_port,
            show_error = not opt.not_show_error,
            debug = opt.debug,
        )


if __name__ == '__main__':
    opt = app_options()
    try:
        load_model(get_checkpoints()[0])
        switch_extractor(default_line_extractor)
        switch_mask_extractor(default_mask_extractor)
        interface = init_interface(opt)
    except Exception as e:
        print(f"Error initializing interface: {e}")
        raise