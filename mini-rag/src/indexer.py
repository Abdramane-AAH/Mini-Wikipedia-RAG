from datasets import load_dataset
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHROMA_PATH,
    COLLECTION_NAME,
    DATASET_CONFIG,
    DATASET_NAME,
    DATASET_SPLIT,
    EMBEDDING_MODEL,
)


def load_corpus(limit: int | None = None) -> list[Document]:
    """Load passages from Hugging Face dataset. If limit is None, loads the entire corpus"""
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    documents = []
    for item in ds:
        doc = Document(
            page_content=item["passage"],
            metadata={"id": str(item["id"])},
        )
        documents.append(doc)

    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into smaller chunks based on config parameters"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def build_vector_store_batched(
    documents: list[Document], 
    chroma_path: str = CHROMA_PATH, 
    batch_size: int = 500
) -> Chroma:
    """Generate embeddings and index documents in ChromaDB in batches to conserve RAM"""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    first_batch = documents[:batch_size]
    vector_store = Chroma.from_documents(
        documents=first_batch,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=chroma_path,
    )
    for i in tqdm(range(batch_size, len(documents), batch_size), desc="Indexing Batches"):
        batch = documents[i : i + batch_size]
        vector_store.add_documents(batch)
        
    return vector_store


if __name__ == "__main__":
    raw_docs = load_corpus(limit=None)
    chunked_docs = chunk_documents(raw_docs)
    build_vector_store_batched(chunked_docs)