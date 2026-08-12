

from unittest.mock import patch
from langchain_core.documents import Document
from src.main import format_docs, generate_answer


def test_format_docs():
    docs = [
        Document(page_content="First chunk of text."),
        Document(page_content="Second chunk of text."),
    ]
    formatted = format_docs(docs)
    assert formatted == "First chunk of text.\n\nSecond chunk of text."


@patch("src.main.build_rag_chain")
@patch("src.main.retrieve_relevant_documents")
def test_generate_answer_structure(mock_retrieve, mock_build_chain):
    mock_retrieve.return_value = [
        Document(page_content="Paris is the capital of France.", metadata={"id": "1"})
    ]
    mock_chain = mock_build_chain.return_value
    mock_chain.invoke.return_value = "Paris"
    result = generate_answer("What is the capital of France?")
    assert result["query"] == "What is the capital of France?"
    assert result["answer"] == "Paris"
    assert len(result["retrieved_docs"]) == 1