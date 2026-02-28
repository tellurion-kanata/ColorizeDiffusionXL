# ColorizeDiffusion XL

<div align="center">

[![arXiv Paper](https://img.shields.io/badge/arXiv-2407.15886%20(base)-B31B1B?style=flat&logo=arXiv)](https://arxiv.org/abs/2401.01456)
[![WACV 2025](https://img.shields.io/badge/WACV%202025-v1-0CA4A5?style=flat&logo=Semantic%20Web)](https://openaccess.thecvf.com/content/WACV2025/html/Yan_ColorizeDiffusion_Improving_Reference-Based_Sketch_Colorization_with_Latent_Diffusion_Model_WACV_2025_paper.html)
[![CVPR 2025](https://img.shields.io/badge/CVPR%202025-v1.5-0CA4A5?style=flat&logo=Semantic%20Web)](https://arxiv.org/abs/2502.19937)
[![arXiv v2 Paper](https://img.shields.io/badge/arXiv-2504.06895%20(v2)-B31B1B?style=flat&logo=arXiv)](https://arxiv.org/abs/2504.06895)
[![Model Weights](https://img.shields.io/badge/Hugging%20Face-Model%20Weights-FF9D00?style=flat&logo=Hugging%20Face)](https://huggingface.co/tellurion/ColorizeDiffusionXL/tree/main)
[![Demo](https://img.shields.io/badge/Hugging%20Face-Demo-FF9D00?style=flat&logo=Hugging%20Face)](https://huggingface.co/spaces/tellurion/ColorizeDiffusion)
[![License](https://img.shields.io/badge/License-CC--BY--NC--SA%204.0-4CAF50?style=flat&logo=Creative%20Commons)](LICENSE)

</div>

SDXL-based implementation of [ColorizeDiffusion](https://github.com/tellurion-kanata/colorizeDiffusion), a reference-based sketch colorization framework built on Stable Diffusion.
This repository contains the XL architecture (1024px) with enhanced embedding guidance for character colorization and geometry disentanglement.
For the base SD2.1 implementation (512/768px), refer to the [original repository](https://github.com/tellurion-kanata/colorizeDiffusion).


## Getting Started

---

```bash
conda env create -f environment.yml
conda activate hf
```


## User Interface

---

```bash
python -u app.py
```

The default server address is [http://localhost:7860](http://localhost:7860).

#### Inference options

| Option                  | Description                                                                                |
|:------------------------|:-------------------------------------------------------------------------------------------|
| Low-level injection     | Enable low-level feature injection for backgrounds.                                        |
| Attention injection     | Noised low-level feature injection, 2x inference time.                                     |
| Reference guidance scale| Classifier-free guidance scale for the reference image.                                    |
| Reference strength      | Decrease to increase semantic fidelity to sketch inputs.                                   |
| Foreground strength     | Reference strength for the foreground region.                                              |
| Background strength     | Reference strength for the background region.                                              |
| Sketch guidance scale   | Classifier-free guidance scale for the sketch image, suggested 1.                          |
| Sketch strength         | Control scale of the sketch condition.                                                     |
| Background factor       | Controls how background region is blended.                                                 |
| Merging scale           | Scale for merging foreground and background.                                               |
| Preprocessor            | Sketch preprocessing. **Extract** is suggested for complicated pencil drawings.            |
| Line extractor          | Line extractors used when preprocessor is **Extract**.                                     |

Text manipulation is deactivated by default. To activate:
```bash
python -u app.py -manipulate
```

Use `--full` to expose additional advanced controls (per-level sketch strengths, cross-attention scales, injection fidelity, etc.).
Refer to the [base repository](https://github.com/tellurion-kanata/colorizeDiffusion) for details on manipulation options.


## Training

---

Our implementation is based on Accelerate and DeepSpeed.
Before starting a training, first collect data and organize your training dataset as follows:
```
[dataset_path]
├── image_list.json    # Optional, for image indexing
├── color/             # Color images (zip archives)
│   ├── 0001.zip
│   │   ├── 10001.png
│   │   ├── 100001.jpg
│   │   └── ...
│   ├── 0002.zip
│   └── ...
├── sketch/            # Sketch images (zip archives)
│   ├── 0001.zip
│   │   ├── 10001.png
│   │   ├── 100001.jpg
│   │   └── ...
│   ├── 0002.zip
│   └── ...
└── mask/              # Mask images (required for adapter training)
    ├── 0001.zip
    │   ├── 10001.png
    │   ├── 100001.jpg
    │   └── ...
    ├── 0002.zip
    └── ...
```

For details of dataset organization, see `data/dataloader.py`.

Training command:
```bash
accelerate launch --config_file [accelerate_config] \
    train.py \
    -n [experiment_name] \
    -d [dataset_path] \
    -bs 16 \
    -nt 4 \
    -cfg configs/training/sdxl-base.yaml \
    -pt [pretrained_model_path] \
    -lr 1e-5 \
    -fm
```
Note that the `batch_size` is micro batch size per GPU. If you run the command on 8 GPUs, the total batch size is 128.
Use `-fm` to fit pretrained weights to the new model architecture.
Refer to `options.py` for full arguments.


## Inference & Validation

---

```bash
# Inference
python inference.py \
    --name inf \
    --dataroot [dataset_path] \
    --batch_size 64 \
    -cfg configs/inference/sdxl.yaml \
    -pt [pretrained_model_path] \
    -gs 5

# Validation (uses random reference images)
python inference.py \
    --name val \
    --dataroot [dataset_path] \
    --batch_size 64 \
    -cfg configs/inference/xl-val.yaml \
    -pt [pretrained_model_path] \
    -gs 5 \
    -val
```

The difference between inference and validation modes is that validation mode uses randomly selected images as reference inputs.
Refer to `options.py` for full arguments.


## Code Reference

---

1. [Stable Diffusion XL](https://github.com/Stability-AI/generative-models)
2. [SD-webui-ControlNet](https://github.com/Mikubill/sd-webui-controlnet)
3. [Stable-Diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
4. [K-diffusion](https://github.com/crowsonkb/k-diffusion)
5. [DeepSpeed](https://github.com/microsoft/DeepSpeed)
6. [sketchKeras-PyTorch](https://github.com/higumax/sketchKeras-pytorch)


## Citation

---

```bibtex
@article{2024arXiv240101456Y,
       author = {{Yan}, Dingkun and {Yuan}, Liang and {Wu}, Erwin and {Nishioka}, Yuma and {Fujishiro}, Issei and {Saito}, Suguru},
        title = "{ColorizeDiffusion: Adjustable Sketch Colorization with Reference Image and Text}",
      journal = {arXiv e-prints},
         year = {2024},
          doi = {10.48550/arXiv.2401.01456},
}

@InProceedings{Yan_2025_WACV,
    author    = {Yan, Dingkun and Yuan, Liang and Wu, Erwin and Nishioka, Yuma and Fujishiro, Issei and Saito, Suguru},
    title     = {ColorizeDiffusion: Improving Reference-Based Sketch Colorization with Latent Diffusion Model},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
    year      = {2025},
    pages     = {5092-5102}
}

@article{2025arXiv250219937Y,
    author = {{Yan}, Dingkun and {Wang}, Xinrui and {Li}, Zhuoru and {Saito}, Suguru and {Iwasawa}, Yusuke and {Matsuo}, Yutaka and {Guo}, Jiaxian},
    title = "{Image Referenced Sketch Colorization Based on Animation Creation Workflow}",
    journal = {arXiv e-prints},
    year = {2025},
    doi = {10.48550/arXiv.2502.19937},
}

@article{yan2025colorizediffusionv2enhancingreferencebased,
      title={ColorizeDiffusion v2: Enhancing Reference-based Sketch Colorization Through Separating Utilities},
      author={Dingkun Yan and Xinrui Wang and Yusuke Iwasawa and Yutaka Matsuo and Suguru Saito and Jiaxian Guo},
      year={2025},
      journal = {arXiv e-prints},
      doi = {10.48550/arXiv.2504.06895},
}
```
