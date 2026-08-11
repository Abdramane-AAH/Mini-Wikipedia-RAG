import os
from dotenv import load_dotenv

load_dotenv()

"""centralized configuration file to keep hyperparameters and paths organized"""

# Dataset
DATASET_NAME = "rag-datasets/rag-mini-wikipedia"
DATASET_CONFIG = "text-corpus"
DATASET_SPLIT = "passages"

# Vector Store & Embeddings
CHROMA_PATH = "data/chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "mini_wikipedia"

# Ollama LLM Config
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = "llama3.2" 