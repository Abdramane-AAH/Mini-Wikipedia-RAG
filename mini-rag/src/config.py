import os
from dotenv import load_dotenv

load_dotenv()

"""centralized configuration file to keep hyperparameters and paths organized"""

# Dataset
DATASET_NAME = "rag-datasets/rag-mini-wikipedia"
DATASET_CONFIG = "text-corpus"
DATASET_SPLIT = "passages"

# Chunking Hyperparameters
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Vector Store & Embeddings
CHROMA_PATH = "data/chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "mini_wikipedia"

# Retriever Hyperparameters
TOP_K = 3

# Ollama LLM Config
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = "llama3.2" 
LLM_TEMPERATURE = 0.0