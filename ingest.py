"""
PDF Ingestion Script
Extracts text and images from PDFs and stores them in ChromaDB for retrieval.
"""

import os
import fitz  # PyMuPDF
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
import hashlib
import json

# Configuration
CUSTOMER_DATA_DIR = Path(__file__).parent.parent / "customer_data"
EXTRACTED_IMAGES_DIR = Path(__file__).parent / "extracted_images"
CHROMA_DB_DIR = Path(__file__).parent / "chroma_db"
CHUNK_SIZE = 1000  # characters per chunk
CHUNK_OVERLAP = 200  # overlap between chunks


def extract_text_from_pdf(pdf_path: Path) -> list[dict]:
    """Extract text from each page of a PDF."""
    doc = fitz.open(pdf_path)
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()

        if text.strip():  # Only include pages with text
            pages.append({
                "page_num": page_num + 1,
                "text": text,
                "source": pdf_path.name
            })

    doc.close()
    return pages


def extract_images_from_pdf(pdf_path: Path, output_dir: Path) -> dict[int, list[str]]:
    """Extract images from PDF and save them. Returns mapping of page_num to image paths."""
    doc = fitz.open(pdf_path)
    page_images = {}

    # Create a safe filename prefix from the PDF name
    pdf_prefix = pdf_path.stem.replace(" ", "_").replace("-", "_")

    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images()
        page_images[page_num + 1] = []

        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)

                # Convert CMYK to RGB if needed
                if pix.n - pix.alpha > 3:
                    pix = fitz.Pixmap(fitz.csRGB, pix)

                # Skip very small images (likely icons/decorations)
                if pix.width < 50 or pix.height < 50:
                    continue

                # Generate unique filename
                img_filename = f"{pdf_prefix}_p{page_num + 1}_img{img_index}.png"
                img_path = output_dir / img_filename

                pix.save(str(img_path))
                page_images[page_num + 1].append(str(img_path))

            except Exception as e:
                print(f"  Warning: Could not extract image {img_index} from page {page_num + 1}: {e}")
                continue

    doc.close()
    return page_images


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at a sentence or paragraph boundary
        if end < len(text):
            # Look for paragraph break
            newline_pos = text.rfind("\n\n", start, end)
            if newline_pos > start + chunk_size // 2:
                end = newline_pos
            else:
                # Look for sentence break
                for punct in [". ", "! ", "? "]:
                    punct_pos = text.rfind(punct, start, end)
                    if punct_pos > start + chunk_size // 2:
                        end = punct_pos + 1
                        break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks


def process_pdfs():
    """Main function to process all PDFs and store in ChromaDB."""
    print("=" * 60)
    print("Customer Data Ingestion")
    print("=" * 60)

    # Ensure directories exist
    EXTRACTED_IMAGES_DIR.mkdir(exist_ok=True)
    CHROMA_DB_DIR.mkdir(exist_ok=True)

    # Initialize ChromaDB with persistent storage
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

    # Use sentence-transformers for embeddings
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Delete existing collection if it exists (fresh start)
    try:
        client.delete_collection("customer_products")
        print("Deleted existing collection")
    except:
        pass

    # Create collection
    collection = client.create_collection(
        name="customer_products",
        embedding_function=embedding_fn,
        metadata={"description": "Customer product data from PDFs"}
    )

    # Find all PDFs
    pdf_files = list(CUSTOMER_DATA_DIR.glob("*.pdf"))
    print(f"\nFound {len(pdf_files)} PDF files to process")

    all_documents = []
    all_metadatas = []
    all_ids = []

    for pdf_path in pdf_files:
        print(f"\nProcessing: {pdf_path.name}")

        # Extract text
        pages = extract_text_from_pdf(pdf_path)
        print(f"  Extracted text from {len(pages)} pages")

        # Extract images
        page_images = extract_images_from_pdf(pdf_path, EXTRACTED_IMAGES_DIR)
        total_images = sum(len(imgs) for imgs in page_images.values())
        print(f"  Extracted {total_images} images")

        # Process each page
        for page_data in pages:
            page_num = page_data["page_num"]
            text = page_data["text"]
            source = page_data["source"]

            # Get images for this page
            images = page_images.get(page_num, [])

            # Chunk the text
            chunks = chunk_text(text)

            for chunk_idx, chunk in enumerate(chunks):
                # Generate unique ID
                doc_id = hashlib.md5(f"{source}_{page_num}_{chunk_idx}_{chunk[:50]}".encode()).hexdigest()

                all_documents.append(chunk)
                all_metadatas.append({
                    "source": source,
                    "page": page_num,
                    "chunk_index": chunk_idx,
                    "images": json.dumps(images),  # Store as JSON string
                    "has_images": len(images) > 0
                })
                all_ids.append(doc_id)

    # Add all documents to collection
    print(f"\nAdding {len(all_documents)} chunks to ChromaDB...")

    # Add in batches (ChromaDB has limits)
    batch_size = 100
    for i in range(0, len(all_documents), batch_size):
        end_idx = min(i + batch_size, len(all_documents))
        collection.add(
            documents=all_documents[i:end_idx],
            metadatas=all_metadatas[i:end_idx],
            ids=all_ids[i:end_idx]
        )

    print(f"\nIngestion complete!")
    print(f"  Total chunks indexed: {len(all_documents)}")
    print(f"  Database location: {CHROMA_DB_DIR}")
    print(f"  Images location: {EXTRACTED_IMAGES_DIR}")

    # Print sample query test
    print("\n" + "=" * 60)
    print("Testing retrieval...")
    print("=" * 60)

    results = collection.query(
        query_texts=["grinding wheel for stainless steel"],
        n_results=3
    )

    print("\nSample query: 'grinding wheel for stainless steel'")
    print("Top 3 results:")
    for i, (doc, metadata) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        print(f"\n{i+1}. Source: {metadata['source']}, Page: {metadata['page']}")
        print(f"   Preview: {doc[:150]}...")


if __name__ == "__main__":
    process_pdfs()
