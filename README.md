# Vitoom Nunchaku

[中文](README_CN.md)

Load [tonera/vitoom-nunchaku](https://huggingface.co/tonera/vitoom-nunchaku) SVDQuant checkpoints in ComfyUI. The DiT is ComfyUI's own model, with quantized linear layers replaced by the nunchaku wheel. Sampling, VAE, and reference images stay on native ComfyUI nodes.

Covers FLUX.1, FLUX.2 Klein, FLUX.3 Image Klein, Qwen-Image 2.1, Z-Image-Turbo, Anima (W8A8), and SDXL.

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

| Quantized model | Quantized text encoder |
| --- | --- |
| FLUX.1: [dev](https://huggingface.co/nunchaku-ai/nunchaku-flux.1-dev), [depth](https://huggingface.co/nunchaku-ai/nunchaku-flux.1-depth-dev), [canny](https://huggingface.co/nunchaku-ai/nunchaku-flux.1-canny-dev), [fill](https://huggingface.co/nunchaku-ai/nunchaku-flux.1-fill-dev), [Kontext](https://huggingface.co/nunchaku-ai/nunchaku-flux.1-kontext-dev), [Krea](https://huggingface.co/nunchaku-ai/nunchaku-flux.1-krea-dev) | [`awq-int4-flux.1-t5xxl`](https://huggingface.co/nunchaku-ai/nunchaku-t5), plus a standard CLIP-L |
| [FLUX.2 Klein 4B](https://huggingface.co/tonera/FLUX.2-klein-4B-Nunchaku) | [`svdq-int4-Qwen3-4B-text-Nunchaku`](https://huggingface.co/tonera/Qwen3-4B-text-Nunchaku) |
| [FLUX.2 Klein 9B](https://huggingface.co/tonera/FLUX.2-klein-9B-Nunchaku) | [`svdq-int4-Qwen3-text-Nunchaku`](https://huggingface.co/tonera/Qwen3-text-Nunchaku) |
| [FLUX.3 Image Klein](https://huggingface.co/tonera/flux-3-image-klein-nunchaku) | [`svdq-int4-Qwen3-VL-4B-Instruct-Nunchaku`](https://huggingface.co/tonera/Qwen3-VL-4B-Instruct-Nunchaku) |
| [Qwen-Image 2.1](https://huggingface.co/tonera/Qwen-Image-2.1-Nunchaku) | [`svdq-int4-Qwen3-VL-8B-Instruct-Nunchaku`](https://huggingface.co/tonera/Qwen3-VL-8B-Instruct-Nunchaku) |
| [Z-Image-Turbo](https://huggingface.co/nunchaku-ai/nunchaku-z-image-turbo) | [`svdq-int4-Qwen3-4B-text-Nunchaku`](https://huggingface.co/tonera/Qwen3-4B-text-Nunchaku). Set the loader target to Z-Image |
| [Anima](https://huggingface.co/tonera/Anima-Turbo-v1.1-Nunchaku) | ComfyUI's own Qwen3-0.6B |
| [SDXL](https://huggingface.co/collections/tonera/sdxl-nunchaku) | ComfyUI's own CLIP-L and CLIP-G |

## Use

Quantized models can be downloaded from [huggingface.co/tonera](https://huggingface.co/tonera) or [modelscope.cn/profile/tonera](https://modelscope.cn/profile/tonera).

Drag a workflow from `example_workflows/` onto the canvas, or open it from the workflow templates. Connect **Vitoom Nunchaku DiT Loader** to `KSampler`. Connect the text encoder's `CLIP` output to the matching encode node, and the VAE to the native VAE loader.

| Node | Output |
| --- | --- |
| Vitoom Nunchaku DiT Loader | `MODEL` |
| Vitoom Nunchaku Text Encoder | `CLIP` for FLUX.3, FLUX.2 Klein, and Qwen-Image 2.1. Set target to Z-Image for Z-Image-Turbo |
| Vitoom Nunchaku Flux CLIP | `CLIP` for FLUX.1, from Nunchaku T5 plus CLIP-L |
| Vitoom Nunchaku LoRA Loader | `MODEL`, and the node can be chained |
| Vitoom Nunchaku Prompt Upsampler | Expanded prompt string |
| Vitoom Nunchaku Model Info | Checkpoint precision, rank, and whether it matches the GPU |

Quantized layers stay on the GPU as a whole and do not take part in dynamic VRAM offloading.

## Acknowledgments

- [nunchaku](https://github.com/nunchux-ai/nunchaku): a high-performance inference engine.
- [deepcompressor](https://github.com/nunchux-ai/deepcompressor): a model compression toolbox.
- [fluxsd.com](https://fluxsd.com/): a site for images, video, posters, comics, and picture books.
- [createpptx.com](https://createpptx.com/): a site for high-quality editable PowerPoint files.
