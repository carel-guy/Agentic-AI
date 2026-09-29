"""
Ingests data/faq.csv into the 'faq' Chroma collection.
Run once (or whenever the CSV changes): python ingest_faq.py
"""
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import pandas as pd
from langchain_core.documents import Document
from config import DATA_DIR
from vector_store import store_documents

COLLECTION  = "faq"
CSV_PATH = DATA_DIR / "faq.csv"


def load_faq_documents(csv_path: str) -> list[Document]:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    required = {"id", "question", "answer", "category"}
    if not required.issubset(df.columns):
        raise ValueError(f"FAQ CSV is missing columns: {sorted(required - set(df.columns))}")
    if df.empty or any(df[column].str.strip().eq("").any() for column in required):
        raise ValueError("FAQ CSV contains no entries or has empty required fields.")
    if df["id"].duplicated().any():
        raise ValueError("FAQ IDs must be unique.")
    docs = []
    for _, row in df.iterrows():
        content = f"Q: {row['question']}\nA: {row['answer']}"
        docs.append(Document(
            page_content=content,
            metadata={"source": "faq", "category": row["category"], "faq_id": str(row["id"])},
        ))
    return docs


def main():
    print("Loading FAQ documents...")
    docs = load_faq_documents(CSV_PATH)
    print(f"  {len(docs)} FAQ entries loaded.")

    print(f"Embedding and storing in Chroma collection '{COLLECTION}'...")
    ids = [f"faq:{doc.metadata['faq_id']}" for doc in docs]
    count = store_documents(COLLECTION, docs, ids)
    print(f"  Done. {count} vectors stored.")


if __name__ == "__main__":
    main()
