from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    TOP_K,
)


def get_vector_store(chroma_path: str = CHROMA_PATH) -> Chroma:
    """Load an existing ChromaDB vector store instance"""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=chroma_path,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


def retrieve_relevant_documents(
    query: str, k: int = TOP_K, chroma_path: str = CHROMA_PATH
) -> list[Document]:
    """Retrieve top-k relevant documents from ChromaDB for a given query"""
    vector_store = get_vector_store(chroma_path=chroma_path)
    results = vector_store.similarity_search(query, k=k)
    return results


if __name__ == "__main__":
    sample_query = "Who was the first president of the United States?"
    retrieved_docs = retrieve_relevant_documents(sample_query)