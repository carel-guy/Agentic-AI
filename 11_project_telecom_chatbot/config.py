"""Shared paths and models for ingestion and retrieval."""
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
CHROMA_DIR = str(PROJECT_DIR / "chroma_store")

# Process settings take priority, then this project's .env, then the root .env.
load_dotenv(PROJECT_DIR / ".env")
load_dotenv(PROJECT_DIR.parent / ".env")

# Same model and default encoding settings as 4_rag_basics/rag.ipynb.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "qwen/qwen3.8-27b").strip()


@lru_cache(maxsize=1)
def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def require_groq_key():
    if not os.getenv("GROQ_API_KEY", "").strip():
        raise ValueError("Add GROQ_API_KEY to the repository or chatbot .env file.")
    if not CHAT_MODEL:
        raise ValueError("GROQ_CHAT_MODEL must contain a valid Groq model ID.")
