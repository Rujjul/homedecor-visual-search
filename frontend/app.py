import streamlit as st
import requests
from PIL import Image
import io

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Home Décor Visual Search",
    page_icon="🏠",
    layout="wide"
)

# ── API config ─────────────────────────────────────────────────────────────────
API_URL = "http://localhost:8000"

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🏠 Home Décor Visual Search")
st.markdown("Upload any home décor image to find visually similar products.")
st.divider()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider(
        "Number of results",
        min_value=4,
        max_value=20,
        value=10,
        step=2
    )
    st.divider()
    st.markdown("**How it works**")
    st.markdown("""
    1. Upload a photo
    2. CLIP encodes it into a 512-d vector
    3. FAISS finds the most similar products
    4. Results ranked by visual similarity
    """)
    st.divider()

    # Health check
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.status_code == 200:
            st.success("API connected ✓")
        else:
            st.error("API error")
    except:
        st.error("API not reachable — is FastAPI running?")

# ── Upload section ─────────────────────────────────────────────────────────────
col_upload, col_preview = st.columns([1, 1])

with col_upload:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a home décor image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload any home décor photo — furniture, lighting, rugs, etc."
    )

with col_preview:
    if uploaded_file:
        st.subheader("Your Image")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

# ── Search ─────────────────────────────────────────────────────────────────────
if uploaded_file:
    st.divider()

    with st.spinner("🔍 Searching catalog..."):
        # Reset file pointer before sending
        uploaded_file.seek(0)
        image_bytes = uploaded_file.read()

        try:
            response = requests.post(
                f"{API_URL}/search",
                files={"file": (uploaded_file.name, image_bytes, uploaded_file.type)},
                params={"top_k": top_k},
                timeout=30
            )

            if response.status_code == 200:
                data     = response.json()
                results  = data["results"]

                st.subheader(f"Top {len(results)} Similar Products")
                st.caption(f"Searched {9998:,} home décor products")

                # ── Results grid ───────────────────────────────────────────────
                # Display in rows of 5 columns
                cols_per_row = 5
                for row_start in range(0, len(results), cols_per_row):
                    row_results = results[row_start : row_start + cols_per_row]
                    cols        = st.columns(cols_per_row)

                    for col, product in zip(cols, row_results):
                        with col:
                            # Product image from Amazon CDN
                            try:
                                st.image(
                                    product["image_url"],
                                    use_container_width=True
                                )
                            except:
                                st.image(
                                    "https://via.placeholder.com/200x200?text=No+Image",
                                    use_container_width=True
                                )

                            # Similarity badge
                            score_pct = product["similarity_score"] * 100
                            if score_pct >= 90:
                                st.success(f"Match: {score_pct:.1f}%")
                            elif score_pct >= 75:
                                st.warning(f"Match: {score_pct:.1f}%")
                            else:
                                st.info(f"Match: {score_pct:.1f}%")

                            # Product details
                            st.markdown(
                                f"**{product['title'][:60]}...**"
                                if len(product['title']) > 60
                                else f"**{product['title']}**"
                            )
                            st.markdown(f"£{product['price']:.2f}")
                            st.markdown(
                                f"⭐ {product['stars']} "
                                f"({product['reviews']:,} reviews)"
                            )
                            st.markdown(f"*{product['category']}*")
                            st.link_button(
                                "View on Amazon",
                                product["product_url"],
                                use_container_width=True
                            )

            else:
                st.error(f"Search failed: {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Make sure FastAPI is running on port 8000.")
        except Exception as e:
            st.error(f"Error: {str(e)}")