"""Check sources by default; opt in to embeddings, retrieval, or a Groq request."""
import argparse
import ast
import json
import math
from importlib.metadata import version

from config import CHAT_MODEL, CHROMA_DIR, EMBED_MODEL, PROJECT_DIR, get_embeddings, require_groq_key


def check_sources():
    from ingest_faq import CSV_PATH, load_faq_documents
    from ingest_pdf import load_guide_documents
    from ingest_tickets import DB_PATH, load_ticket_documents

    for path in PROJECT_DIR.glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for package in (
        "langchain", "langchain-core", "langchain-chroma", "langchain-groq",
        "langchain-huggingface", "chromadb", "sentence-transformers", "pandas",
        "python-dotenv", "streamlit", "pypdf", "langchain-community", "langchain-text-splitters",
    ):
        print(f"  {package}: {version(package)}")

    reference = PROJECT_DIR.parent / "4_rag_basics" / "rag.ipynb"
    if reference.is_file():
        notebook = json.loads(reference.read_text(encoding="utf-8"))
        reference_models = []
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            for node in ast.walk(ast.parse("".join(cell["source"]))):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "HuggingFaceEmbeddings":
                    reference_models.extend(
                        keyword.value.value for keyword in node.keywords
                        if keyword.arg == "model_name" and isinstance(keyword.value, ast.Constant)
                    )
        if EMBED_MODEL not in reference_models:
            raise ValueError("Embedding model differs from 4_rag_basics/rag.ipynb.")
        print("Embedding model matches 4_rag_basics.")

    sources = {
        "faq": load_faq_documents(CSV_PATH),
        "tickets": load_ticket_documents(DB_PATH),
        "guides": load_guide_documents(),
    }
    for name, docs in sources.items():
        if not docs or any(not doc.page_content.strip() for doc in docs):
            raise ValueError(f"Source '{name}' is empty or contains blank documents.")
        print(f"Source {name}: {len(docs)} documents")
    return sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", action="store_true", help="Load MiniLM and encode sample text; does not ingest.")
    parser.add_argument("--retrieval", action="store_true", help="Check collection contents and search the existing index.")
    parser.add_argument("--live", action="store_true", help="Also verify Groq model access and generate a RAG answer.")
    args = parser.parse_args()

    print(f"Chat model: {CHAT_MODEL}")
    print(f"Embedding model: {EMBED_MODEL}")
    print(f"Vector database: {CHROMA_DIR}")
    sources = check_sources()

    if args.embeddings:
        vector = get_embeddings().embed_query("How do I activate international roaming?")
        if len(vector) != 384 or not all(math.isfinite(value) for value in vector):
            raise ValueError("Expected 384 finite embedding values from MiniLM.")
        print("Embedding check: OK (384 dimensions)")

    question = "How do I activate international roaming?"
    if args.retrieval or args.live:
        from retriever import build_retriever, open_vector_stores

        id_fields = {"faq": ("faq", "faq_id"), "tickets": ("ticket", "ticket_id"), "guides": ("guide", "chunk_index")}
        for name, store in open_vector_stores().items():
            prefix, field = id_fields[name]
            expected = {f"{prefix}:{doc.metadata[field]}" for doc in sources[name]}
            actual = set(store.get(include=[])["ids"])
            if actual != expected:
                raise ValueError(f"Collection '{name}' is incomplete or has obsolete IDs; rerun its ingestion script.")
            print(f"Collection {name}: OK ({len(actual)} vectors)")
        docs = build_retriever().invoke(question)
        if {doc.metadata.get("source") for doc in docs} != {"faq", "ticket", "guide"}:
            raise ValueError("Retrieval did not return documents from all three sources.")
        print(f"Retrieval check: OK ({len(docs)} documents)")

    if args.live:
        from groq import Groq
        from rag_chain import build_chain

        require_groq_key()
        with Groq(timeout=30, max_retries=0) as client:
            if CHAT_MODEL not in {model.id for model in client.models.list().data}:
                raise ValueError(f"Groq does not list model '{CHAT_MODEL}' for this account.")
        print("Groq model access: OK")
        answer = build_chain().invoke(question)
        if not answer.strip():
            raise ValueError("Groq returned an empty answer.")
        print("Live RAG answer:")
        print(answer)

    print("All requested checks passed. No ingestion was performed.")


if __name__ == "__main__":
    main()
