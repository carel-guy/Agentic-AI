"""
Builds a merged retriever across all three Chroma collections:
  - faq     : FAQ entries (no chunking — 1 row = 1 doc)
  - tickets : resolved support tickets (no chunking — 1 ticket = 1 doc)
  - guides  : PDF guide chunks (RecursiveCharacterTextSplitter applied at ingest)
"""
from pathlib import Path

from chromadb.errors import NotFoundError
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document

from config import CHROMA_DIR, EMBED_MODEL, get_embeddings

COLLECTIONS = ("faq", "tickets", "guides")
INGEST_HELP = "Run ingest_faq.py, ingest_tickets.py and ingest_pdf.py before starting the chatbot."


def open_vector_stores() -> dict[str, Chroma]:
    if not (Path(CHROMA_DIR) / "chroma.sqlite3").is_file():
        raise ValueError(f"Vector database not found. {INGEST_HELP}")
    stores = {}
    for name in COLLECTIONS:
        try:
            store = Chroma(
                collection_name=name,
                persist_directory=CHROMA_DIR,
                create_collection_if_not_exists=False,
            )
        except NotFoundError as exc:
            raise ValueError(f"Collection '{name}' is missing. {INGEST_HELP}") from exc
        if not store.get(limit=1, include=[])["ids"]:
            raise ValueError(f"Collection '{name}' is empty. {INGEST_HELP}")
        stored_model = (store._collection.metadata or {}).get("embedding_model")
        if stored_model and stored_model != EMBED_MODEL:
            raise ValueError(f"Collection '{name}' uses a different embedding model: {stored_model}")
        stores[name] = store
    return stores


def build_retriever(
    k_faq: int = 3,
    k_tickets: int = 3,
    k_guides: int = 3,
) -> RunnableLambda:
    limits = {"faq": k_faq, "tickets": k_tickets, "guides": k_guides}
    if any(k < 1 for k in limits.values()):
        raise ValueError("Each retrieval limit must be at least 1.")
    stores = open_vector_stores()
    embeddings = get_embeddings()

    def retrieve(query: str) -> list[Document]:
        vector = embeddings.embed_query(query)
        return [
            doc
            for name, store in stores.items()
            for doc in store.similarity_search_by_vector(vector, k=limits[name])
        ]

    return RunnableLambda(retrieve)
