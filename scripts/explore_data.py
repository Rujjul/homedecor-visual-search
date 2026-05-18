import pandas as pd
import os

#load the csv
csv_path = "data/raw/amz_uk_processed_data.csv"
print("Loading dataset :")
df = pd.read_csv(csv_path)

#.shape gives us(total rows, total columns)
print("\n"+"="*50)
print("BASIC INFO")
print(f"Total products : {df.shape[0]:,}")
print(f"Total columns : {df.shape[1]}")

#column names + data types
print("\n" + "="*50)
print("COLUMNS + DATA TYPES")
print("="*50)
for col in df.columns:
    print(f" {col:<25} -> {df[col].dtype}")

#first 3 rows
print("\n"+"="*50)
print("FIRST 3 ROWS")
print("="*50)
for i, row in df.head(3).iterrows():
    print(f"\n--- Prouct {i+1} ---")
    for col in df.columns:
        print(f" {col}: {row[col]}")
        
#missing values
print("\n"+"="*50)
print("MISSING VALUES")
print("="*50)
missing = df.isnull().sum()
missing_pct = (missing/len(df) * 100).round(2)
for col in df.columns:
    print(f" {col:<25} -> {missing[col]:>7,} missing ({missing_pct[col]}%)")

#categories
print("\n" + "="*50)
print("ALL UNIQUE CATEGORIES")
print("="*50)
# Find the category column — check common names
category_col = None
for possible in ['category', 'Category', 'main_category', 'categoryName']:
    if possible in df.columns:
        category_col = possible
        break
if category_col:
    cats = df[category_col].value_counts()
    print(f"\nFound category column: '{category_col}'")
    print(f"Total unique categories: {len(cats)}\n")
    for cat, count in cats.items():
        print(f"  {count:>8,} products → {cat}")
else:
    print("No obvious category column found.")
    print("Check column names above and look for category-like columns.")  
    

# ── 7. Image URL check ─────────────────────────────────────────────────────────
# Visual search only works if we have image URLs
# Let's verify they exist and look valid
print("\n" + "="*50)
print("IMAGE URL CHECK")
print("="*50)

img_col = None
for possible in ['imgUrl', 'image_url', 'imageUrl', 'img_url', 'image']:
    if possible in df.columns:
        img_col = possible
        break

if img_col:
    valid_urls = df[img_col].dropna()
    print(f"Image column found: '{img_col}'")
    print(f"Total with image URLs: {len(valid_urls):,}")
    print(f"\nSample URLs:")
    for url in valid_urls.head(3):
        print(f"  {url}")
else:
    print("No image URL column found — check column names above.")

# ── 8. Price check ─────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("PRICE STATS")
print("="*50)
price_col = None
for possible in ['price', 'Price', 'final_price']:
    if possible in df.columns:
        price_col = possible
        break

if price_col:
    price_data = pd.to_numeric(df[price_col], errors='coerce').dropna()
    print(f"Price column: '{price_col}'")
    print(f"  Min   : £{price_data.min():.2f}")
    print(f"  Max   : £{price_data.max():.2f}")
    print(f"  Median: £{price_data.median():.2f}")
    print(f"  Mean  : £{price_data.mean():.2f}")

print("\n" + "="*50)
print("EXPLORATION COMPLETE")
print("="*50)      