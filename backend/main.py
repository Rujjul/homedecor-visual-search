from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn

from backend.search import search_by_image

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Home Décor Visual Search API",
    description="Upload an image, get visually similar home décor products",
    version="1.0.0"
)

# CORS — allows Streamlit (on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Response schema ────────────────────────────────────────────────────────────
# Pydantic model — defines exactly what each result looks like
# FastAPI uses this for automatic validation and documentation
class Product(BaseModel):
    product_id       : str
    title            : str
    category         : str
    price            : float
    stars            : float
    reviews          : int
    image_url        : str
    product_url      : str
    similarity_score : float

class SearchResponse(BaseModel):
    query_received : bool
    results_count  : int
    results        : List[Product]

# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "running", "message": "Home Décor Visual Search API"}

@app.get("/health")
def health():
    return {"status": "healthy"}

# ── Search endpoint ────────────────────────────────────────────────────────────
@app.post("/search", response_model=SearchResponse)
async def search(
    file: UploadFile = File(...),   # ... means required
    top_k: int = 10
):
    """
    Upload an image file → returns top_k visually similar products.

    - Accepts: JPEG, PNG, WEBP
    - Returns: list of products with title, price, image, similarity score
    """

    # Validate file type
    allowed_types = {"image/jpeg", "image/png", "image/webp"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file.content_type} not supported. Use JPEG, PNG, or WEBP."
        )

    # Validate file size — reject anything over 10MB
    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum size is 10MB."
        )

    # Run search
    try:
        results = search_by_image(image_bytes, top_k=top_k)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )

    return SearchResponse(
        query_received=True,
        results_count=len(results),
        results=results
    )

# ── Run directly ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
    