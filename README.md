# Home Décor Visual Search

Aesthetic-aware visual search engine for home décor products.
Upload any image → find visually similar products from a 200k+ catalog.

## Tech Stack
- **CLIP** — image embeddings (OpenAI via Hugging Face)
- **FAISS** — vector similarity search (Facebook)
- **FastAPI** — search API
- **Streamlit** — frontend UI

## Setup
\```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
\```

## Phases
- [x] Phase 1: Visual search (CLIP + FAISS)
- [ ] Phase 2: Aesthetic classifier
- [ ] Phase 3: Hybrid recommender
- [ ] Phase 4: AI shopping assistant
- [ ] Phase 5: Room coherence check