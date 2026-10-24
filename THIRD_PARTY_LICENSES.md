# Third-party licenses

This repository does not redistribute any third-party code, weights or images. The code downloads them from their original sources at run time. Their licenses still apply to anyone who downloads them, and to the trained weights we release (Section 6 of the README), because those weights are derived from them.

## Pretrained backbones

| Backbone | Source | License | Our checkpoints derived from it |
|---|---|---|---|
| DINOv3 ViT-H+/16, DINOv3 ViT-L/16 | Meta (`timm/vit_huge_plus_patch16_dinov3.lvd1689m`, `facebook/dinov3-vitl16-pretrain-lvd1689m`) | DINOv3 License, https://ai.meta.com/resources/models-and-libraries/dinov3-license | F8c, G, F1, and the two CMASK out-of-fold models (not released) |
| DINOv2 ViT-L/14 with registers | Meta (`dinov2_vitl14_reg4_pretrain.pth`, dl.fbaipublicfiles.com) | Apache-2.0 | L2c, L2 seed 42, L2 seed 2 |
| C-RADIOv4-H | NVIDIA (`nvidia/C-RADIOv4-H`) | NVIDIA Open Model License, https://developer.download.nvidia.com/licenses/nvidia-open-model-license-agreement-june-2024.pdf | R |

When the trained weights are redistributed:

- weights derived from DINOv3 must be accompanied by a copy of the DINOv3 License and remain under its terms, which include an acknowledgement of the DINO materials in publications and a list of prohibited uses;
- weights derived from C-RADIOv4-H remain under the NVIDIA Open Model License;
- the X2F-CMASK and C4 packages contain several of these checkpoints, so all of the licenses above apply to them.

## Training data

| Data | License or terms |
|---|---|
| UPAR annotations | CC BY-NC-SA 3.0, https://github.com/speckean/upar_dataset |
| PA-100K images | Terms of the PA-100K authors |
| PETA images | Research use only, no commercial use |
| Market-1501 images and attributes | Terms of the Market-1501 authors |

Because the weights were trained on these data, we release them for **non-commercial research use only**.

## Software

| Package | License |
|---|---|
| PyTorch, torchvision, NumPy, pandas | BSD-3-Clause |
| timm, transformers, huggingface_hub, safetensors | Apache-2.0 |
| Pillow | MIT-CMU (HPND) |
| gdown, Jupyter | MIT / BSD-3-Clause |

This summary is not legal advice. The license texts linked above are authoritative.
