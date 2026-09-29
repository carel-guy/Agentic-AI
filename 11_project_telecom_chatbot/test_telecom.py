"""Offline regression tests: no Groq calls, model downloads, or vector ingestion."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

import config
import ingest_faq
import ingest_pdf
import ingest_tickets
import retriever
import vector_store


class TelecomTests(unittest.TestCase):
    def test_data_paths_are_anchored_to_project(self):
        for path in (ingest_faq.CSV_PATH, ingest_tickets.DB_PATH, ingest_pdf.PDF_PATH):
            self.assertTrue(Path(path).is_absolute())
            self.assertEqual(Path(path).parent, config.DATA_DIR)
        self.assertEqual(Path(config.CHROMA_DIR).parent, config.PROJECT_DIR)

    def test_faq_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "faq.csv"
            path.write_text("id,question,answer,category\n1,Q,A,test\n1,Q2,A2,test\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unique"):
                ingest_faq.load_faq_documents(path)

    def test_missing_ticket_db_is_not_created(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.db"
            with self.assertRaises(ingest_tickets.sqlite3.OperationalError):
                ingest_tickets.load_ticket_documents(path)
            self.assertFalse(path.exists())

    def test_missing_index_does_not_load_model_or_create_store(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing-index"
            with patch.object(retriever, "CHROMA_DIR", str(path)), patch.object(retriever, "get_embeddings") as embed, patch.object(retriever, "Chroma") as store:
                with self.assertRaisesRegex(ValueError, "ingest_faq.py"):
                    retriever.build_retriever()
                embed.assert_not_called()
                store.assert_not_called()
            self.assertFalse(path.exists())

    def test_reingestion_reuses_ids_and_removes_obsolete_documents(self):
        documents = [Document(page_content="new answer")]
        store = MagicMock()
        store._collection.metadata = {"embedding_model": config.EMBED_MODEL}
        store.get.side_effect = [{"ids": ["faq:1", "faq:2"]}, {"ids": ["faq:1"]}]
        with patch.object(vector_store, "Chroma", return_value=store), patch.object(vector_store, "get_embeddings"):
            self.assertEqual(vector_store.store_documents("faq", documents, ["faq:1"]), 1)
        store.add_documents.assert_called_once_with(documents=documents, ids=["faq:1"])
        store.delete.assert_called_once_with(ids=["faq:2"])

    def test_failed_embedding_does_not_delete_existing_vectors(self):
        store = MagicMock()
        store._collection.metadata = {"embedding_model": config.EMBED_MODEL}
        store.get.return_value = {"ids": ["faq:old"]}
        store.add_documents.side_effect = RuntimeError("embedding failed")
        with patch.object(vector_store, "Chroma", return_value=store), patch.object(vector_store, "get_embeddings"):
            with self.assertRaisesRegex(RuntimeError, "embedding failed"):
                vector_store.store_documents("faq", [Document(page_content="new")], ["faq:1"])
        store.delete.assert_not_called()

    def test_invalid_documents_do_not_open_a_store(self):
        with patch.object(vector_store, "Chroma") as store:
            with self.assertRaisesRegex(ValueError, "No documents"):
                vector_store.store_documents("faq", [], [])
            with self.assertRaisesRegex(ValueError, "unique"):
                vector_store.store_documents("faq", [Document(page_content="a"), Document(page_content="b")], ["same", "same"])
            store.assert_not_called()

    def test_retrieval_embeds_query_once_for_three_collections(self):
        stores = {name: MagicMock() for name in retriever.COLLECTIONS}
        for name, store in stores.items():
            store.similarity_search_by_vector.return_value = [Document(page_content=name)]
        embeddings = MagicMock()
        embeddings.embed_query.return_value = [0.1, 0.2]
        with patch.object(retriever, "open_vector_stores", return_value=stores), patch.object(retriever, "get_embeddings", return_value=embeddings):
            result = retriever.build_retriever().invoke("roaming")
        self.assertEqual(len(result), 3)
        embeddings.embed_query.assert_called_once_with("roaming")
        for store in stores.values():
            store.similarity_search_by_vector.assert_called_once_with([0.1, 0.2], k=3)


if __name__ == "__main__":
    unittest.main()
