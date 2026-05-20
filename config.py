"""
============================================================
config.py — Centralized Configuration
============================================================
All project-wide settings live here. Load once, import everywhere.
Reads API keys from .env file and exposes constants used across
the entire RAG pipeline.
============================================================
"""

import os
import sys
from dotenv import load_dotenv

# ── Load environment variables from .env file ──────────────
load_dotenv()

# ── API Keys ───────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# ── Model Configuration ───────────────────────────────────
# LLM model for response generation (streaming supported)
LLM_MODEL = "gemini-2.5-flash-lite"

# Embedding model for vector generation
EMBEDDING_MODEL = "gemini-embedding-2"

# Embedding output dimensionality (gemini-embedding-2 supports 768, 1536, 3072)
# Using 768 for storage efficiency with minimal quality loss (MRL support)
EMBEDDING_DIMENSION = 768

# ── ChromaDB Configuration ────────────────────────────────
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "finance-rag-chatbot")
CHROMA_PERSIST_DIR = os.getenv(
    "CHROMA_PERSIST_DIR",
    os.path.join(os.path.dirname(__file__), "chroma_db"),
)

# ── Chunking Configuration ────────────────────────────────
# Characters per chunk — 500 is a good balance for finance text
CHUNK_SIZE = 500

# Overlap between consecutive chunks — preserves context at boundaries
CHUNK_OVERLAP = 100

# ── Retrieval Configuration ───────────────────────────────
# Number of top results to retrieve from vector store
DEFAULT_TOP_K = 5

# ── Conversation Memory ──────────────────────────────────
# Maximum number of conversation turns to keep in memory
MAX_MEMORY_TURNS = 10

# ── Batch Sizes ───────────────────────────────────────────
# Max vectors per vector store upsert call
UPSERT_BATCH_SIZE = 100

# Max texts per embedding API call
EMBEDDING_BATCH_SIZE = 100

# ── File Paths ────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
DATASET_FILE = os.path.join(DATA_DIR, "finance_dataset.txt")
LOG_FILE = os.path.join(LOGS_DIR, "rag_finance.log")

# ── Retry Configuration ──────────────────────────────────
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0  # seconds — exponential backoff base


def validate_env():
    """
    Validate that all required environment variables are set.
    Exits with a helpful message if any are missing.
    """
    missing = []
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY":
        missing.append("GOOGLE_API_KEY")
    if missing:
        print("\n╔══════════════════════════════════════════════════════╗")
        print("║          ⚠️  MISSING API KEYS DETECTED              ║")
        print("╠══════════════════════════════════════════════════════╣")
        for key in missing:
            print(f"║  ❌  {key:<47} ║")
        print("╠══════════════════════════════════════════════════════╣")
        print("║  1. Copy .env.example to .env                      ║")
        print("║  2. Add your API keys to the .env file             ║")
        print("║  3. Re-run the application                         ║")
        print("╚══════════════════════════════════════════════════════╝\n")
        sys.exit(1)
