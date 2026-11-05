# Vitoom Nunchaku

[中文](README_CN.md)

Load [tonera/vitoom-nunchaku](https://huggingface.co/tonera/vitoom-nunchaku) SVDQuant checkpoints in ComfyUI. The DiT is ComfyUI's own model, with quantized linear layers replaced by the nunchaku wheel. Sampling, VAE, and reference images stay on native ComfyUI nodes.

Covers FLUX.1, FLUX.2 Klein, FLUX.3 Image Klein, Qwen-Image 2.1, Z-Image, Anima (W8A8), and SDXL.

## Requirements

- ComfyUI 0.44 or newer, started with `--enable-manager`
- Python 3.11, 3.12, or 3.13
- A CUDA build of PyTorch on Linux x86_64 or Windows x64

The nunchaku 1.4.1 wheels published today match PyTorch 2.14 and CUDA 13.0. The installer picks the file that matches the Python running ComfyUI. If PyTorch or CUDA does not match, the custom node still installs, but the wheel does not, and loading a model fails.

RTX 50-series GPUs only accept W4A4 checkpoints whose names contain `svdq-fp4`.

## Install

1. Open ComfyUI and click **Manage Extensions** in the top-right corner.
2. Search for `Vitoom Nunchaku` and click **Install**.
3. Restart ComfyUI.

The install step also installs the matching nunchaku wheel and the Python packages in `requirements.txt`. If Hugging Face cannot be reached, the installer retries through `hf-mirror.com`.

## Models

| File | Directory |
| --- | --- |
| DiT checkpoint (`svdq-fp4`, `svdq-int4`, or `svdq-int8`) | `ComfyUI/models/diffusion_models` |
| Nunchaku Qwen text encoder, FLUX.1 T5 | `ComfyUI/models/text_encoders` or `~/weights` |
| LoRA | `ComfyUI/models/loras` |
| VAE | `ComfyUI/models/vae` |

Qwen-Image 2.1 uses the native encoder node **Text Encode Qwen Image 2.1** and the VAE `qwen_image_2.1_vae_bf16.safetensors`. FLUX.1 also needs a CLIP-L file in `models/text_encoders`.

## Use

Drag a workflow from `example_workflows/` onto the canvas, or open it from the workflow templates. Connect **Vitoom Nunchaku DiT Loader** to `KSampler`. Connect the text encoder to the native encode node, and the VAE to the native VAE loader.

| Node | Output |
| --- | --- |
| Vitoom Nunchaku DiT Loader | `MODEL` |
| Vitoom Nunchaku Text Encoder | `CLIP` for FLUX.3, FLUX.2 Klein, and Qwen-Image 2.1 |
| Vitoom Nunchaku Flux CLIP | `CLIP` for FLUX.1, from Nunchaku T5 plus CLIP-L |
| Vitoom Nunchaku LoRA Loader | `MODEL`, and the node can be chained |
| Vitoom Nunchaku Prompt Upsampler | Expanded prompt string |
| Vitoom Nunchaku Model Info | Checkpoint precision, rank, and whether it matches the GPU |

Quantized layers stay on the GPU as a whole and do not take part in dynamic VRAM offloading.
