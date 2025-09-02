# DINOv3 Model Size Investigation

This repository contains code to investigate the output tensor sizes of DINOv3 models.

## Model Output Sizes

The following table shows the output tensor dimensions for each DINOv3 model when processing a single image (batch size = 1):

| Model Name | Feature Dimension |
|------------|-------------------|
| dinov3_vits16 | 384 |
| dinov3_vits16plus | 384 |
| dinov3_vitb16 | 768 |
| dinov3_vitl16 | 1024 |
| dinov3_vith16plus | 1280 |
| dinov3_vit7b16 | 4096 |
| dinov3_convnext_tiny | 768 |
| dinov3_convnext_small | 768 |
| dinov3_convnext_base | 1024 |
| dinov3_convnext_large | 1536 |

## Prerequisites

1. Download DINOv3 weights from the official Meta AI link: https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/
2. Clone the DINOv3 repository
3. Update the `models_config` dictionary with your actual weight file paths