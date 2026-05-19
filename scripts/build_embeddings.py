import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────────
IMAGE_DIR      = Path("data/images")
EMBEDDINGS_DIR = Path("embeddings")
BATCH_SIZE     = 32      # process 32 images at once — faster than one by one
MODEL_NAME     = "openai/clip-vit-base-patch32"
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

# ── Device setup ───────────────────────────────────────────────────────────────
# Use GPU if available, otherwise CPU
# For most laptops this will be CPU — still fast enough
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ── Load CLIP ──────────────────────────────────────────────────────────────────
# First run downloads ~600MB model from Hugging Face — cached after that
# Subsequent runs load instantly from cache
print(f"Loading CLIP model ({MODEL_NAME})...")
model     = CLIPModel.from_pretrained(MODEL_NAME).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model.eval()  # set to evaluation mode — disables dropout, saves memory
print("Model loaded.\n")

# ── Load sampled catalog ───────────────────────────────────────────────────────
df = pd.read_csv("data/processed/home_decor_sampled.csv")
df['product_id'] = df['product_id'].astype(str)
print(f"Products to encode: {len(df):,}")

# ── Verify images exist ────────────────────────────────────────────────────────
# Only encode products where the image file actually exists
df['image_path'] = df['product_id'].apply(
    lambda pid: IMAGE_DIR / f"{pid}.jpg"
)
df = df[df['image_path'].apply(lambda p: p.exists())].reset_index(drop=True)
print(f"Images found on disk: {len(df):,}")

# ── Batch encoding ─────────────────────────────────────────────────────────────
# Why batches?
# Encoding one image at a time: model loads/unloads for each image → slow
# Encoding in batches: model processes 32 images in one forward pass → fast
# BATCH_SIZE=32 is a sweet spot for CPU — increase to 64 if you have >16GB RAM

all_embeddings = []   # will collect (batch_size, 512) tensors
all_ids        = []   # will collect product_id strings

print(f"\nGenerating embeddings in batches of {BATCH_SIZE}...")

# Process in chunks of BATCH_SIZE
for batch_start in tqdm(
    range(0, len(df), BATCH_SIZE),
    desc="Encoding",
    unit="batch"
):
    batch_df = df.iloc[batch_start : batch_start + BATCH_SIZE]

    # ── Load and preprocess images in this batch ───────────────────────────────
    images     = []
    valid_ids  = []

    for _, row in batch_df.iterrows():
        try:
            # PIL opens the image
            # .convert("RGB") ensures consistent 3-channel format
            # Some images are RGBA (4 channels) or grayscale — CLIP needs RGB
            img = Image.open(row['image_path']).convert("RGB")
            images.append(img)
            valid_ids.append(row['product_id'])
        except Exception as e:
            # Corrupted or unreadable image — skip silently
            pass

    if not images:
        continue

    # ── CLIP preprocessing ─────────────────────────────────────────────────────
    # processor handles:
    # 1. Resize to 224×224 (CLIP's required input size)
    # 2. Normalize pixel values to [-1, 1]
    # 3. Convert to PyTorch tensor
    # return_tensors="pt" means return PyTorch tensors
    inputs = processor(
        images=images,
        return_tensors="pt",
        padding=True
    ).to(device)

    # ── Forward pass ───────────────────────────────────────────────────────────
    # torch.no_grad() tells PyTorch not to compute gradients
    # We're doing inference (not training) so gradients are wasted memory
    with torch.no_grad():
        features = model.get_image_features(**inputs)

    # Handle both tensor and ModelOutput return types
    if not isinstance(features, torch.Tensor):
        features = features.image_embeds \
            if hasattr(features, 'image_embeds') \
            else features.pooler_output

    # L2 normalise — makes cosine similarity = dot product
    features = features / features.norm(dim=-1, keepdim=True)

    # ── L2 normalisation ───────────────────────────────────────────────────────
    # Normalise each vector to unit length (magnitude = 1)
    # Why? Makes cosine similarity = simple dot product
    # FAISS can then use faster dot product search instead of full cosine
    features = features / features.norm(dim=-1, keepdim=True)

    # Move from GPU/CPU tensor to numpy array
    embeddings_np = features.cpu().numpy()

    all_embeddings.append(embeddings_np)
    all_ids.extend(valid_ids)

# ── Stack all batches into one matrix ─────────────────────────────────────────
# np.vstack turns list of (32, 512) arrays into one (N, 512) array
final_embeddings = np.vstack(all_embeddings).astype(np.float32)
final_ids        = np.array(all_ids)

print(f"\nEmbedding matrix shape : {final_embeddings.shape}")
print(f"  → {final_embeddings.shape[0]:,} products")
print(f"  → {final_embeddings.shape[1]} dimensions per product")

# ── Save to disk ───────────────────────────────────────────────────────────────
# .npy is numpy's native binary format
# Saves and loads in under a second even for large matrices
np.save(EMBEDDINGS_DIR / "product_embeddings.npy", final_embeddings)
np.save(EMBEDDINGS_DIR / "product_ids.npy",        final_ids)

print(f"\nSaved:")
print(f"  embeddings/product_embeddings.npy  "
      f"({final_embeddings.nbytes / 1024 / 1024:.1f} MB)")
print(f"  embeddings/product_ids.npy")

# ── Sanity check ───────────────────────────────────────────────────────────────
# Reload and verify everything saved correctly
loaded = np.load(EMBEDDINGS_DIR / "product_embeddings.npy")
ids    = np.load(EMBEDDINGS_DIR / "product_ids.npy", allow_pickle=True)

print(f"\nSanity check:")
print(f"  Loaded shape   : {loaded.shape}")
print(f"  First ID       : {ids[0]}")
print(f"  First vector   : {loaded[0][:5]} ...")
print(f"  Vector magnitude: {np.linalg.norm(loaded[0]):.4f} (should be 1.0)")

print(f"\n{'='*50}")
print(f"EMBEDDING GENERATION COMPLETE")
print(f"{'='*50}")