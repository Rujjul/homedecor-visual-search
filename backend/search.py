import numpy as np
import faiss
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import io

# ── Config ─────────────────────────────────────────────────────────────────────
EMBEDDINGS_DIR = Path("embeddings")
MODEL_NAME     = "openai/clip-vit-base-patch32"
TOP_K          = 10   # number of results to return

# ── Load everything once at module import ──────────────────────────────────────
# This runs once when FastAPI starts — not on every request
# Keeps search fast — no reloading model or index per query
print("Loading CLIP model...")
device    = "cuda" if torch.cuda.is_available() else "cpu"
model     = CLIPModel.from_pretrained(MODEL_NAME).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model.eval()
print(f"CLIP loaded on {device}")

print("Loading FAISS index...")
index = faiss.read_index(str(EMBEDDINGS_DIR / "faiss_index.bin"))
print(f"FAISS index loaded — {index.ntotal:,} vectors")

print("Loading product catalog...")
# Load id mapping to convert FAISS int IDs back to product_id strings
df_ids = pd.read_csv(EMBEDDINGS_DIR / "id_mapping.csv")
id_map = dict(zip(df_ids['faiss_id'], df_ids['product_id'].astype(str)))

# Load sampled catalog for product metadata lookup
df_catalog = pd.read_csv("data/processed/home_decor_sampled.csv")
df_catalog['product_id'] = df_catalog['product_id'].astype(str)
df_catalog = df_catalog.set_index('product_id')
print(f"Catalog loaded — {len(df_catalog):,} products")

# ── Core search function ───────────────────────────────────────────────────────
def search_by_image(image_bytes: bytes, top_k: int = TOP_K) -> list[dict]:
    """
    Takes raw image bytes (from uploaded file)
    Returns list of top_k similar products as dicts

    Flow:
    image bytes → PIL Image → CLIP processor → CLIP encoder
    → query vector → FAISS search → product IDs → metadata lookup
    → list of result dicts
    """

    # ── 1. Load image ──────────────────────────────────────────────────────────
    # io.BytesIO wraps bytes so PIL can read them like a file
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # ── 2. Preprocess + encode ─────────────────────────────────────────────────
    inputs = processor(
        images=image,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        features = model.get_image_features(**inputs)

    # Handle both tensor and ModelOutput return types
    if not isinstance(features, torch.Tensor):
        features = features.image_embeds \
            if hasattr(features, 'image_embeds') \
            else features.pooler_output

    # L2 normalise — must match how catalog embeddings were normalised
    features = features / features.norm(dim=-1, keepdim=True)
    query_vector = features.cpu().numpy().astype(np.float32)

    # ── 3. FAISS search ────────────────────────────────────────────────────────
    # Returns (1, top_k) arrays of scores and IDs
    scores, faiss_ids = index.search(query_vector, top_k)

    # ── 4. Build results ───────────────────────────────────────────────────────
    results = []
    for score, faiss_id in zip(scores[0], faiss_ids[0]):
        # Convert FAISS int ID back to product_id string
        product_id = id_map.get(int(faiss_id))
        if product_id is None:
            continue

        # Look up metadata from catalog
        if product_id not in df_catalog.index:
            continue

        row = df_catalog.loc[product_id]

        results.append({
            "product_id"      : product_id,
            "title"           : str(row['title']),
            "category"        : str(row['categoryName']),
            "price"           : float(row['price']),
            "stars"           : float(row['stars']),
            "reviews"         : int(row['reviews']),
            "image_url"       : str(row['imgUrl']),
            "product_url"     : str(row['productURL']),
            "similarity_score": float(score),
        })

    return results