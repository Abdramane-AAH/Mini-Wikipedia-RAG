import shutil
import tempfile
from langchain_core.documents import Document
from src.indexer import build_vector_store
from src.retriever import retrieve_relevant_documents


def test_retrieve_relevant_documents():
    temp_dir = tempfile.mkdtemp()
    try:
        sample_docs = [
            Document(page_content="Paris is the capital and most populous city of France.", metadata={"id": "1"}),
            Document(page_content="Berlin is the capital of Germany.", metadata={"id": "2"}),
            Document(page_content="Ottawa is the capital city of Canada.", metadata={"id": "3"}),
        ]
        build_vector_store(sample_docs, chroma_path=temp_dir)
        results = retrieve_relevant_documents(query="Tell me about France", k=2, chroma_path=temp_dir)

        assert len(results) == 2
        assert isinstance(results[0], Document)
        assert "Paris" in results[0].page_content
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_retriever_respects_top_k():
    temp_dir = tempfile.mkdtemp()
    try:
        sample_docs = [
            Document(page_content=f"Document number {i}", metadata={"id": str(i)})
            for i in range(5)
        ]
        build_vector_store(sample_docs, chroma_path=temp_dir)

        results = retrieve_relevant_documents(query="Document", k=3, chroma_path=temp_dir)
        assert len(results) == 3
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)