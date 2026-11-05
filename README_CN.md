# Vitoom Nunchaku

[English](README.md)

在 ComfyUI 里加载 [tonera/vitoom-nunchaku](https://huggingface.co/tonera/vitoom-nunchaku) 的 SVDQuant 检查点。DiT 用 ComfyUI 自带的模型，被量化的线性层换成 nunchaku wheel。采样、VAE 和参考图仍用 ComfyUI 原生节点。

覆盖 FLUX.1、FLUX.2 Klein、FLUX.3 Image Klein、Qwen-Image 2.1、Z-Image、Anima（W8A8）和 SDXL。

## 环境

- ComfyUI 0.44 或更新版本，启动时带 `--enable-manager`
- Python 3.11、3.12 或 3.13
- Linux x86_64 或 Windows x64 上的 CUDA 版 PyTorch

当前发布的 nunchaku 1.4.1 wheel 对应 PyTorch 2.14 和 CUDA 13.0。安装时按正在运行 ComfyUI 的 Python 选择文件。PyTorch 或 CUDA 对不上时，插件目录仍会留下，但 wheel 不会装上，加载模型会失败。

RTX 50 系列只能用文件名里带 `svdq-fp4` 的 W4A4 检查点。

## 安装

1. 打开 ComfyUI，点右上角 **管理扩展功能**。
2. 搜索 `Vitoom Nunchaku`，点 **安装**。
3. 重启 ComfyUI。

安装时会一并装上匹配的 nunchaku wheel，以及 `requirements.txt` 里的 Python 包。连不上 Hugging Face 时，会自动改走 `hf-mirror.com`。

## 模型

| 文件 | 目录 |
| --- | --- |
| DiT 检查点（`svdq-fp4`、`svdq-int4` 或 `svdq-int8`） | `ComfyUI/models/diffusion_models` |
| Nunchaku Qwen 文本编码器、FLUX.1 的 T5 | `ComfyUI/models/text_encoders` 或 `~/weights` |
| LoRA | `ComfyUI/models/loras` |
| VAE | `ComfyUI/models/vae` |

Qwen-Image 2.1 用原生编码节点 **Text Encode Qwen Image 2.1**，VAE 用 `qwen_image_2.1_vae_bf16.safetensors`。FLUX.1 还需要 `models/text_encoders` 里的 CLIP-L。

## 使用

把 `example_workflows/` 里的工作流拖进画布，或从工作流模板打开。**Vitoom Nunchaku DiT Loader** 接到 `KSampler`。文本编码器接到原生编码节点，VAE 接到原生 VAE 加载器。

| 节点 | 输出 |
| --- | --- |
| Vitoom Nunchaku DiT Loader | `MODEL` |
| Vitoom Nunchaku Text Encoder | `CLIP`，用于 FLUX.3、FLUX.2 Klein 和 Qwen-Image 2.1 |
| Vitoom Nunchaku Flux CLIP | `CLIP`，用于 FLUX.1，由 Nunchaku T5 和 CLIP-L 组成 |
| Vitoom Nunchaku LoRA Loader | `MODEL`，节点可以串联 |
| Vitoom Nunchaku Prompt Upsampler | 扩写后的提示词 |
| Vitoom Nunchaku Model Info | 检查点精度、rank，以及是否匹配当前 GPU |

量化层整块留在 GPU 上，不参与动态显存卸载。
