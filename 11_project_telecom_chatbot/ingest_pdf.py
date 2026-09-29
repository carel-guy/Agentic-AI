"""
Ingests data/telecom_guide.pdf into the 'guides' Chroma collection.
Applies RecursiveCharacterTextSplitter to break the long document into chunks.
Run once (or after regenerating the PDF): python ingest_pdf.py
"""
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import DATA_DIR
from vector_store import store_documents

COLLECTION = "guides"
PDF_PATH = DATA_DIR / "telecom_guide.pdf"

CHUNK_SIZE    = 600
CHUNK_OVERLAP = 100


def load_guide_documents(pdf_path=PDF_PATH):
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    if not pages or not any(page.page_content.strip() for page in pages):
        raise ValueError("The PDF has no extractable text; OCR is required before ingestion.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(pages)

    # Tag each chunk so we know it came from the guide
    for i, chunk in enumerate(chunks):
        chunk.metadata["source"] = "guide"
        chunk.metadata["chunk_index"] = i
    return chunks


def main():
    print("Loading and chunking PDF...")
    chunks = load_guide_documents()
    print(f"  {len(chunks)} chunks produced.")

    print(f"Embedding and storing in Chroma collection '{COLLECTION}'...")
    ids = [f"guide:{doc.metadata['chunk_index']}" for doc in chunks]
    count = store_documents(COLLECTION, chunks, ids)
    print(f"  Done. {count} vectors stored.")


if __name__ == "__main__":
    main()
