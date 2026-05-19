import numpy as np
import faiss
import pandas as pd
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
EMBEDDINGS_DIR = Path("embeddings")
INDEX_PATH     = EMBEDDINGS_DIR / "faiss_index.bin"

# ── Load embeddings ────────────────────────────────────────────────────────────
# These are the .npy files build_embeddings.py saved
# product_embeddings.npy → (9998, 512) float32 matrix
# product_ids.npy        → (9998,) array of product ID strings
print("Loading embeddings...")
embeddings = np.load(EMBEDDINGS_DIR / "product_embeddings.npy")
ids        = np.load(EMBEDDINGS_DIR / "product_ids.npy", allow_pickle=True)

print(f"Embeddings shape : {embeddings.shape}")
print(f"Total products   : {len(ids):,}")
print(f"Dimensions       : {embeddings.shape[1]}")

# ── Verify normalisation ───────────────────────────────────────────────────────
# Embeddings should be L2 normalised (magnitude = 1.0)
# We did this in build_embeddings.py
# Verify it held after saving/loading
sample_magnitude = np.linalg.norm(embeddings[0])
print(f"Sample magnitude : {sample_magnitude:.4f} (should be ~1.0)")

# ── Build FAISS index ──────────────────────────────────────────────────────────
# FAISS has many index types — we use IndexFlatIP
# "Flat"  → exact search, no approximation, checks every vector
# "IP"    → Inner Product (dot product)
# Since vectors are L2 normalised, dot product = cosine similarity
# This is the most accurate index type — good for our catalog size
#
# For 1M+ vectors you'd switch to IndexIVFFlat or HNSW for speed
# At 10k vectors, IndexFlatIP is instant anyway

print("\nBuilding FAISS index...")
dimension = embeddings.shape[1]   # 512

index = faiss.IndexFlatIP(dimension)

# faiss.IndexIDMap wraps the index so we can assign our own IDs
# Without this, FAISS only returns row numbers (0, 1, 2...)
# With this, we can map directly to our product IDs
# We use integer IDs internally — store the string→int mapping separately
index_with_ids = faiss.IndexIDMap(index)

# FAISS needs integer IDs — convert product_id strings to int64
# e.g. "100023" → 100023
int_ids = ids.astype(np.int64)

# Add all vectors to the index
# embeddings must be float32 (already is from build_embeddings.py)
# int_ids must be int64
index_with_ids.add_with_ids(embeddings, int_ids)

print(f"Vectors in index : {index_with_ids.ntotal:,}")

# ── Test the index ─────────────────────────────────────────────────────────────
# Before saving, verify search actually works
# Query with the first product's embedding — top result should be itself
print("\nTesting search...")
test_vector   = embeddings[0:1]   # shape (1, 512) — first product
k             = 5                 # return top 5 results

scores, result_ids = index_with_ids.search(test_vector, k)

print(f"Query product ID : {ids[0]}")
print(f"Top {k} results:")
for rank, (score, rid) in enumerate(zip(scores[0], result_ids[0])):
    print(f"  #{rank+1}  ID: {rid:<8}  similarity: {score:.4f}")

# Top result should be the query itself with score ~1.0
assert result_ids[0][0] == int(ids[0]), "Top result should be query itself"
assert scores[0][0] > 0.99, f"Self similarity should be ~1.0, got {scores[0][0]}"
print("Search test passed ✓")

# ── Save index ─────────────────────────────────────────────────────────────────
faiss.write_index(index_with_ids, str(INDEX_PATH))
print(f"\nSaved: {INDEX_PATH}")
print(f"Index size: {INDEX_PATH.stat().st_size / 1024 / 1024:.1f} MB")

# ── Save ID mapping ────────────────────────────────────────────────────────────
# We need to map from integer FAISS IDs back to product metadata
# Save a simple CSV that maps int_id → product_id string
df_ids = pd.DataFrame({
    'faiss_id'   : int_ids,
    'product_id' : ids
})
df_ids.to_csv(EMBEDDINGS_DIR / "id_mapping.csv", index=False)
print(f"Saved: embeddings/id_mapping.csv")

# ── Final summary ──────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"FAISS INDEX BUILT SUCCESSFULLY")
print(f"{'='*50}")
print(f"Index type    : IndexFlatIP (exact cosine search)")
print(f"Vectors       : {index_with_ids.ntotal:,}")
print(f"Dimensions    : {dimension}")
print(f"Index file    : {INDEX_PATH}")
print(f"\nReady for Step 5 — FastAPI search endpoint")