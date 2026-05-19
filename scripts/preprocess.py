import pandas as pd
import os
print("Starting preprocessing . . . ")

#load raw data
df = pd.read_csv("data/raw/amz_uk_processed_data.csv")
print(f"Raw dataset: {len(df):,} products")

#homedecor categories to keep
HOME_DECOR_CATEGORIES = [
    # Furniture
    "Living Room Furniture",
    "Bedroom Furniture",
    "Hallway Furniture",
    "Bathroom Furniture",
    "Dining Room Furniture",
    "Home Office Furniture",
    "Home Entertainment Furniture",
    "Home Bar Furniture",
    "Furniture & Lighting",
    "Garden Furniture & Accessories",

    # Lighting
    "Indoor Lighting",
    "Outdoor Lighting",
    "String Lights",
    "Lights and switches",
    "Lighting",
    "Bathroom Lighting",

    # Décor & Accessories
    "Handmade Home Décor",
    "Garden Décor",
    "Decorative Home Accessories",
    "Decorative Artificial Flora",
    "Home Fragrance",
    "Signs & Plaques",
    "Photo Frames",
    "Vases",
    "Candles & Holders",
    "Mirrors",
    "Clocks",
    "Doormats",
    "Window Treatments",
    "Curtain & Blind Accessories",

    # Soft furnishings & textiles
    "Cushions & Accessories",
    "Rugs, Pads & Protectors",
    "Bedding Collections",
    "Bedding & Linen",
    "Bathroom Linen",
    "Kitchen Linen",
    "Mattress Pads & Toppers",

    # Storage & organisation
    "Storage & Organisation",
    "Storage & Home Organisation",
    "Kitchen Storage & Organisation",
    "Boxes & Organisers",

    # Kitchen & dining
    "Tableware",
    "Handmade Kitchen & Dining Products",
    "Handmade Home & Kitchen Products",
]

#filter to home decor only
df_decor = df[df['categoryName'].isin(HOME_DECOR_CATEGORIES)].copy()
print(f"After category filter: {len(df_decor):,} products")

#dropping products with missing or invalid images  or broken urls -> even though exploration showed 0 missing
df_decor = df_decor[df_decor['imgUrl'].notna()]
df_decor = df_decor[df_decor['imgUrl'].str.startswith('https://')]
print(f"After image URL filter: {len(df_decor):,} products")

#drop products with missing titles
df_decor = df_decor[df_decor['title'].notna()]
df_decor = df_decor[df_decor['title'].str.strip() != '']
print(f"After title filter: {len(df_decor):,} products")

#filter price outliers
# Realistic home décor: £1 to £5,000
df_decor = df_decor[df_decor['price'] >= 1.0]
df_decor = df_decor[df_decor['price'] <= 5000.0]
print(f"After price filter (£1–£5000): {len(df_decor):,} products")

#remove duplicates
before_dedup = len(df_decor)
df_decor = df_decor.drop_duplicates(subset='asin', keep='first')
print(f"After deduplication: {len(df_decor):,} products "
      f"(removed {before_dedup - len(df_decor):,} duplicates)")

#reset index
df_decor = df_decor.reset_index(drop=True)

#add a clean product_id column
df_decor['product_id'] = df_decor.index.astype(str).str.zfill(6)

#reorder for clarity
df_final = df_decor[[
    'product_id',
    'asin',
    'title',
    'categoryName',
    'price',
    'stars',
    'reviews',
    'isBestSeller',
    'boughtInLastMonth',
    'imgUrl',
    'productURL',
]]

#save the processed dataset
output_path = "data/processed/home_decor_products.csv"
df_final.to_csv(output_path, index=False)
print(f"\nSaved to: {output_path}")

#final summary
print("\n" + "="*50)
print("PREPROCESSING COMPLETE")
print("="*50)
print(f"Total products saved : {len(df_final):,}")
print(f"\nCategory breakdown:")
for cat, count in df_final['categoryName'].value_counts().items():
    print(f"  {count:>7,} → {cat}")

print(f"\nPrice stats after cleaning:")
print(f"  Min   : £{df_final['price'].min():.2f}")
print(f"  Max   : £{df_final['price'].max():.2f}")
print(f"  Mean   : £{df_final['price'].mean():.2f}")
print(f"  Median: £{df_final['price'].median():.2f}")

print(f"\nSample of final data:")
df_final['title_short'] = df_final['title'].str[:60] + '...'
print(df_final[['product_id', 'title_short', 'categoryName', 'price']].head(5).to_string(index=False))