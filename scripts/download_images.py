import pandas as pd
import requests
import os
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Config ─────────────────────────────────────────────────────────────────────
NUM_PRODUCTS = 10_000
IMAGE_DIR    = Path("data/images")
MAX_WORKERS  = 16
TIMEOUT      = 10
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# ── Load + sample ──────────────────────────────────────────────────────────────
print("Loading catalog...")
df = pd.read_csv("data/processed/home_decor_products.csv")

# Fix: use plain string index as product_id — no zero padding
# This ensures product_id always matches the image filename on disk
df['product_id'] = df.index.astype(str)

print(f"Total available: {len(df):,} products")

# ── Stratified sampling ────────────────────────────────────────────────────────
print(f"Stratified sampling {NUM_PRODUCTS:,} products...")
df_sample = (
    df.groupby('categoryName', group_keys=False)
    .apply(lambda x: x.sample(
        min(len(x), max(1, int(NUM_PRODUCTS * len(x) / len(df)))),
        random_state=42
    ))
)

if len(df_sample) < NUM_PRODUCTS:
    remaining = df.drop(df_sample.index)
    topup = remaining.sample(
        min(NUM_PRODUCTS - len(df_sample), len(remaining)),
        random_state=42
    )
    df_sample = pd.concat([df_sample, topup])

df_sample = df_sample.head(NUM_PRODUCTS).reset_index(drop=True)

# ── Skip already downloaded ────────────────────────────────────────────────────
pending = [
    (str(row['product_id']), row['imgUrl'])
    for _, row in df_sample.iterrows()
    if not (IMAGE_DIR / f"{row['product_id']}.jpg").exists()
]
already_done = len(df_sample) - len(pending)
print(f"Already downloaded : {already_done:,}")
print(f"Remaining          : {len(pending):,}")

# ── Download function ──────────────────────────────────────────────────────────
def download_image(args):
    product_id, img_url = args
    save_path = IMAGE_DIR / f"{product_id}.jpg"

    try:
        response = requests.get(img_url, timeout=TIMEOUT, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return product_id, True
        return product_id, False
    except Exception:
        return product_id, False

# ── Parallel download ──────────────────────────────────────────────────────────
success_ids = []
failed_ids  = []

if pending:
    print(f"\nDownloading with {MAX_WORKERS} parallel workers...\n")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(download_image, args): args[0]
            for args in pending
        }
        with tqdm(total=len(pending), desc="Downloading", unit="img") as pbar:
            for future in as_completed(futures):
                product_id, success = future.result()
                if success:
                    success_ids.append(product_id)
                else:
                    failed_ids.append(product_id)
                pbar.update(1)
else:
    print("\nAll images already downloaded — skipping download step.")

# ── Save sampled catalog ───────────────────────────────────────────────────────
# Collect every image stem currently on disk
disk_stems = set(
    p.stem for p in IMAGE_DIR.iterdir()
    if p.name.lower().endswith('.jpg')
)
print(f"\nImages found on disk : {len(disk_stems)}")

# Match disk stems against sampled product_ids
df_sample['product_id'] = df_sample['product_id'].astype(str)
df_downloaded = df_sample[df_sample['product_id'].isin(disk_stems)].copy()
df_downloaded = df_downloaded.reset_index(drop=True)

# Save — this is the definitive catalog for CLIP encoding
df_downloaded.to_csv("data/processed/home_decor_sampled.csv", index=False)

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"DOWNLOAD COMPLETE")
print(f"{'='*50}")
print(f"Already on disk    : {already_done:,}")
print(f"Newly downloaded   : {len(success_ids):,}")
print(f"Failed             : {len(failed_ids):,}")
print(f"Total matched      : {len(df_downloaded):,}")
print(f"Success rate       : {len(df_downloaded) / NUM_PRODUCTS * 100:.1f}%")
print(f"\nTop categories:")
for cat, cnt in df_downloaded['categoryName'].value_counts().head(8).items():
    print(f"  {cnt:>5,} → {cat}")
print(f"\nSaved: data/processed/home_decor_sampled.csv")