# Customer Q&A Application

A RAG-powered question-answering system for product catalogs using Claude AI.

## Quick Start

### 1. Install Dependencies

```bash
cd customer-qa-app
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set Up Environment

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### 3. Index the PDFs

```bash
python ingest.py
```

This will:
- Extract text from all PDFs in `../customer_data/`
- Extract images from the PDFs
- Create embeddings and store in ChromaDB

### 4. Run the Application

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Deploy to Streamlit Cloud

1. Push this folder to a GitHub repository

2. Go to [share.streamlit.io](https://share.streamlit.io)

3. Connect your GitHub repo

4. Add secrets in Streamlit Cloud dashboard:
   - Go to App Settings > Secrets
   - Add: `ANTHROPIC_API_KEY = "your_key_here"`

5. For the PDFs and database:
   - Option A: Include `chroma_db/` in your repo (run ingest locally first)
   - Option B: Use cloud storage (S3, GCS) for PDFs

## Project Structure

```
customer-qa-app/
├── app.py              # Streamlit frontend
├── ingest.py           # PDF processing script
├── requirements.txt    # Python dependencies
├── .env.example        # Environment template
├── chroma_db/          # Vector database (generated)
└── extracted_images/   # Images from PDFs (generated)
```

## How It Works

1. **Ingestion**: PDFs are processed to extract text and images
2. **Embedding**: Text chunks are embedded using sentence-transformers
3. **Storage**: Embeddings stored in ChromaDB for fast retrieval
4. **Query**: User questions are embedded and similar chunks retrieved
5. **Answer**: Claude generates answers using retrieved context
6. **Display**: Answer shown with relevant product images
