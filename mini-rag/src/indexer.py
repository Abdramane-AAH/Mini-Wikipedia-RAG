from datasets import load_dataset
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHROMA_PATH,
    COLLECTION_NAME,
    DATASET_CONFIG,
    DATASET_NAME,
    DATASET_SPLIT,
    EMBEDDING_MODEL,
)


def load_corpus(limit: int = 100) -> list[Document]:
    """Load passages from Hugging Face dataset and convert to LangChain Documents"""
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)
    documents = []
    subset = dataset.select(range(min(limit, len(dataset))))

    for item in subset:
        doc = Document(
            page_content=item["passage"],
            metadata={"id": str(item["id"]), "title": item.get("title", "")},
        )
        documents.append(doc)

    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into smaller chunks for RAG based on config parameters"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def build_vector_store(
    documents: list[Document], chroma_path: str = CHROMA_PATH
) -> Chroma:
    """Generate embeddings and index documents in a local ChromaDB instance"""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=chroma_path,
    )
    return vector_store


if __name__ == "__main__":
    raw_docs = load_corpus(limit=500)
    chunked_docs = chunk_documents(raw_docs)
    build_vector_store(chunked_docs)