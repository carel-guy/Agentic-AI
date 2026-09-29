"""Synchronize each source with its own Chroma collection."""
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import CHROMA_DIR, EMBED_MODEL, get_embeddings


def store_documents(collection: str, documents: list[Document], ids: list[str]) -> int:
    if not documents:
        raise ValueError(f"No documents to ingest into '{collection}'; existing data was kept.")
    if len(ids) != len(documents) or len(set(ids)) != len(ids):
        raise ValueError("Each document must have one unique, stable ID.")

    store = Chroma(
        collection_name=collection,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
        collection_metadata={"embedding_model": EMBED_MODEL},
    )
    stored_model = (store._collection.metadata or {}).get("embedding_model")
    if stored_model and stored_model != EMBED_MODEL:
        raise ValueError(f"Collection '{collection}' uses a different embedding model: {stored_model}")

    previous_ids = set(store.get(include=[])["ids"])
    # Chroma upserts these IDs, so repeated ingestion does not add duplicates.
    store.add_documents(documents=documents, ids=ids)
    obsolete_ids = sorted(previous_ids - set(ids))
    if obsolete_ids:
        store.delete(ids=obsolete_ids)
    return len(store.get(include=[])["ids"])
