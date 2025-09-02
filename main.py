import torch
import gc

REPO_DIR = "./dinov3"
models_config = {
    # DINOv3 ViT models pretrained on web images
    "dinov3_vits16" : "dinov3_vits16_pretrain_lvd1689m.pth",
    "dinov3_vits16plus" : "dinov3_vits16plus_pretrain_lvd1689m.pth",
    "dinov3_vitb16": "dinov3_vitb16_pretrain_lvd1689m.pth",
    "dinov3_vitl16": "dinov3_vitl16_pretrain_lvd1689m.pth",
    "dinov3_vith16plus": "dinov3_vith16plus_pretrain_lvd1689m.pth",
    "dinov3_vit7b16": "dinov3_vit7b16_pretrain_lvd1689m.pth",

    # DINOv3 ConvNeXt models pretrained on web images
    "dinov3_convnext_tiny" : "dinov3_convnext_tiny_pretrain_lvd1689m.pth",
    "dinov3_convnext_small" : "dinov3_convnext_small_pretrain_lvd1689m.pth",
    "dinov3_convnext_base" : "dinov3_convnext_base_pretrain_lvd1689m.pth",
    "dinov3_convnext_large" : "dinov3_convnext_large_pretrain_lvd1689m.pth",
}

device = "cuda" if torch.cuda.is_available() else "cpu"

input_tensor = torch.randn(1, 3, 224, 224).to(device)

results = []

for model_name, model_path in models_config.items():
    model = torch.hub.load(
        REPO_DIR,
        model_name,
        source='local',
        weights=model_path,
    ).to(device)
    output_tensor = model(input_tensor)
    results.append((model_name, model_path, output_tensor.size()))

    print(output_tensor.size())

    del model
    torch.cuda.empty_cache()
    gc.collect()

# results
for model_name, model_path, size in results:
    print(f"{model_name}, {model_path}: {size}")