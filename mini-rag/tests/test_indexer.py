import shutil
import tempfile
from langchain_core.documents import Document
from src.indexer import build_vector_store, chunk_documents, load_corpus


def test_load_corpus():
    limit = 5
    docs = load_corpus(limit=limit)
    assert len(docs) == limit
    assert isinstance(docs[0], Document)
    assert len(docs[0].page_content) > 0
    assert "id" in docs[0].metadata


def test_chunk_documents():
    sample_docs = [
        Document(
            page_content="A " * 400, metadata={"id": "1"}
        )  # 800 characters long
    ]
    chunks = chunk_documents(sample_docs)

    assert len(chunks) > 1
    assert all(len(c.page_content) <= 600 for c in chunks)


def test_build_vector_store():
    temp_dir = tempfile.mkdtemp()
    try:
        sample_docs = [
            Document(page_content="Paris is the capital of France.", metadata={"id": "1"}),
            Document(page_content="Tokyo is the capital of Japan.", metadata={"id": "2"}),
        ]
        vector_store = build_vector_store(sample_docs, chroma_path=temp_dir)
        results = vector_store.similarity_search("France", k=1)
        assert len(results) == 1
        assert "France" in results[0].page_content
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)