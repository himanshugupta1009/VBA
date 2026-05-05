import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

# -------- 1. Load model --------
model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(
    model_name,
    cache_dir="/home/himanshu/.cache/huggingface"
)
processor = CLIPProcessor.from_pretrained(model_name,
    cache_dir="/home/himanshu/.cache/huggingface"
)

# Inspect dimensions
print("vision hidden:", clip_model.vision_model.config.hidden_size)
print("text hidden:", clip_model.text_model.config.hidden_size)
print("projection dim:", clip_model.config.projection_dim)

# Token projections for your VLA
vision_proj = nn.Linear(
    clip_model.vision_model.config.hidden_size,
    clip_model.config.projection_dim,
    bias=False,
)

text_proj = nn.Linear(
    clip_model.text_model.config.hidden_size,
    clip_model.config.projection_dim,
    bias=False,
)

# Initialize from CLIP's learned global projection heads
with torch.no_grad():
    vision_proj.weight.copy_(clip_model.visual_projection.weight)
    text_proj.weight.copy_(clip_model.text_projection.weight)

# -------- 2. Example inputs --------
image = Image.open("example.jpg").convert("RGB")
text = ["pick up the red cup"]

inputs = processor(text=text, images=image, return_tensors="pt", padding=True)

# -------- 3. Forward pass --------
with torch.no_grad():
    outputs = clip_model(**inputs)

# -------- 4. Extract tokens --------
# Vision tokens (includes CLS token at index 0)
vision_tokens = outputs.vision_model_output.last_hidden_state
# Shape: (B, N_patches + 1, D)

# Remove CLS token if you want only patches
vision_patch_tokens = vision_tokens[:, 1:, :]
vision_patch_tokens = vision_proj(vision_patch_tokens)
# Shape: (B, N_patches, D)

# Text tokens
text_tokens = outputs.text_model_output.last_hidden_state
text_tokens = text_proj(text_tokens)
# Shape: (B, N_tokens, D)

# -------- 5. Print shapes --------
print("Vision tokens (with CLS):", vision_tokens.shape)
print("Vision patch tokens:", vision_patch_tokens.shape)
print("Text tokens:", text_tokens.shape)