"""
Customer Q&A Application
A Streamlit app that answers questions about products using RAG with Claude.
"""

import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from anthropic import Anthropic
from pathlib import Path
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
CHROMA_DB_DIR = Path(__file__).parent / "chroma_db"
NUM_RESULTS = 5  # Number of chunks to retrieve

# Page config
st.set_page_config(
    page_title="Product Q&A Assistant",
    page_icon="🔧",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stImage {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 4px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def init_chromadb():
    """Initialize ChromaDB client and collection."""
    if not CHROMA_DB_DIR.exists():
        return None

    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    try:
        collection = client.get_collection(
            name="customer_products",
            embedding_function=embedding_fn
        )
        return collection
    except:
        return None


@st.cache_resource
def init_anthropic():
    """Initialize Anthropic client."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    return Anthropic(api_key=api_key)


def retrieve_context(collection, query: str, n_results: int = NUM_RESULTS) -> tuple[str, list[str], list[dict]]:
    """Retrieve relevant context from ChromaDB."""
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    context_parts = []
    all_images = []
    sources = []

    for i, (doc, metadata) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        source = metadata["source"]
        page = metadata["page"]

        context_parts.append(f"[Source: {source}, Page {page}]\n{doc}")

        # Parse images JSON
        images_json = metadata.get("images", "[]")
        images = json.loads(images_json) if images_json else []
        all_images.extend(images)

        sources.append({
            "source": source,
            "page": page
        })

    context = "\n\n---\n\n".join(context_parts)
    # Remove duplicate images while preserving order
    unique_images = list(dict.fromkeys(all_images))

    return context, unique_images, sources


def get_answer(client: Anthropic, query: str, context: str) -> str:
    """Get answer from Claude using the retrieved context."""
    system_prompt = """You are a helpful product expert assistant. You help customers find the right industrial products (abrasives, grinding wheels, cut-off wheels, fiber discs) based on their needs.

Use the provided context to answer questions accurately. When recommending products:
- Be specific about product names and specifications
- Mention key features and benefits
- If comparing products, highlight key differences
- If you're unsure or the context doesn't contain enough information, say so

Always cite which source document your information comes from."""

    user_prompt = f"""Context from product documentation:

{context}

---

Customer Question: {query}

Please provide a helpful, accurate answer based on the context above."""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_prompt}
        ]
    )

    return message.content[0].text


def main():
    st.title("🔧 Product Q&A Assistant")
    st.markdown("Ask questions about our industrial products catalog")

    # Initialize services
    collection = init_chromadb()
    anthropic_client = init_anthropic()

    # Check if database exists
    if collection is None:
        st.error("⚠️ Product database not found. Please run `python ingest.py` first to index the PDFs.")
        st.code("cd customer-qa-app && python ingest.py", language="bash")
        return

    # Check if API key is set
    if anthropic_client is None:
        st.error("⚠️ ANTHROPIC_API_KEY not found. Please set it in your .env file or environment variables.")
        st.code("export ANTHROPIC_API_KEY=your_key_here", language="bash")
        return

    # Show collection stats
    count = collection.count()
    images_dir = Path(__file__).parent / "extracted_images"
    image_count = len(list(images_dir.glob("*.png"))) if images_dir.exists() else 0

    st.sidebar.markdown("**📊 Database Stats**")
    st.sidebar.markdown(f"Indexed chunks: {count}")
    st.sidebar.markdown(f"Extracted images: {image_count}")

    # Sample questions
    st.sidebar.markdown("---")
    st.sidebar.markdown("**💡 Sample Questions**")

    sample_questions = [
        "What grinding wheel would you recommend for stainless steel?",
        "What are the differences between 3M and Norton cut-off wheels?",
        "Which fiber disc is most cost effective for metal finishing?",
        "What products work best for heavy material removal?",
        "What grit size should I use for surface blending?",
        "Which cut-off wheel lasts longest for cutting steel?",
        "What's the best abrasive for weld removal?",
        "Compare fiber discs vs grinding wheels for rust removal"
    ]

    # Initialize the text input session state if not exists
    if "main_input" not in st.session_state:
        st.session_state.main_input = ""

    # Handle sample question clicks - set the text input value directly
    for q in sample_questions:
        if st.sidebar.button(q, key=f"btn_{q}"):
            st.session_state.main_input = q
            st.rerun()

    # Main input
    st.markdown("---")

    query = st.text_input(
        "Ask a question about our products:",
        placeholder="e.g., What grinding wheel works best for aluminum?",
        key="main_input"
    )

    if st.button("🔍 Search", type="primary"):
        if query:
            with st.spinner("Searching product documentation..."):
                context, images, sources = retrieve_context(collection, query)

            with st.spinner("Generating answer..."):
                answer = get_answer(anthropic_client, query, context)

            # Display answer
            st.markdown("### Answer")
            st.markdown(answer)

            # Display sources
            st.markdown("---")
            st.markdown("### 📚 Sources")

            for src in sources:
                st.markdown(f"- **{src['source']}** (Page {src['page']})")

            # Display relevant images
            if images:
                st.markdown("---")
                st.markdown("### 🖼️ Related Product Images")

                cols = st.columns(min(len(images), 3))

                for idx, img_path in enumerate(images[:6]):
                    if Path(img_path).exists():
                        col_idx = idx % 3
                        with cols[col_idx]:
                            st.image(img_path, use_container_width=True)
                            img_name = Path(img_path).stem
                            st.caption(img_name)


if __name__ == "__main__":
    main()
