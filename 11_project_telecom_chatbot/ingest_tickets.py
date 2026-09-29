"""
Ingests resolved tickets from data/tickets.db into the 'tickets' Chroma collection.
Run once (or after adding new tickets): python ingest_tickets.py
"""
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import sqlite3
from contextlib import closing
from pathlib import Path
from langchain_core.documents import Document
from config import DATA_DIR
from vector_store import store_documents

COLLECTION  = "tickets"
DB_PATH = DATA_DIR / "tickets.db"


def load_ticket_documents(db_path: str) -> list[Document]:
    # Read-only mode also prevents SQLite from creating an empty DB at a wrong path.
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM tickets WHERE status = 'resolved' ORDER BY ticket_id"
        ).fetchall()

    docs = []
    for row in rows:
        required = ("ticket_id", "category", "issue_type", "description", "resolution")
        if any(row[key] is None or not str(row[key]).strip() for key in required):
            raise ValueError("A resolved ticket has missing required fields.")
        # Combine issue description + resolution into a single searchable text block
        content = (
            f"Issue: {row['issue_type']}\n"
            f"Description: {row['description']}\n"
            f"Resolution: {row['resolution']}"
        )
        docs.append(Document(
            page_content=content,
            metadata={
                "source":    "ticket",
                "ticket_id": str(row["ticket_id"]),
                "category":  row["category"],
                "status":    row["status"],
            },
        ))
    return docs


def main():
    print("Loading ticket documents from SQLite...")
    docs = load_ticket_documents(DB_PATH)
    print(f"  {len(docs)} resolved tickets loaded.")

    print(f"Embedding and storing in Chroma collection '{COLLECTION}'...")
    ids = [f"ticket:{doc.metadata['ticket_id']}" for doc in docs]
    count = store_documents(COLLECTION, docs, ids)
    print(f"  Done. {count} vectors stored.")


if __name__ == "__main__":
    main()
