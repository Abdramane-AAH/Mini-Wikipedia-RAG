from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from src.config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    RERANKER_MODEL,
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

def rerank_documents(query: str, docs: list[Document], top_k: int = TOP_K) -> list[Document]:
    """Re-rank retrieved documents using a Cross-Encoder model"""
    if not docs:
        return []
    reranker = CrossEncoder(RERANKER_MODEL)
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)
    scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_docs[:top_k]]

def retrieve_relevant_documents(
    query: str, k: int = TOP_K, chroma_path: str = CHROMA_PATH, use_reranker: bool = True
) -> list[Document]:
    """Retrieve top-k relevant documents with optional re-ranking"""
    vector_store = get_vector_store(chroma_path=chroma_path)
    initial_k = k * 3 if use_reranker else k
    initial_docs = vector_store.similarity_search(query, k=initial_k)
    if use_reranker:
        return rerank_documents(query, initial_docs, top_k=k)
    return initial_docs


if __name__ == "__main__":
    sample_query = "Who was the first president of the United States?"
    retrieved_docs = retrieve_relevant_documents(sample_query)