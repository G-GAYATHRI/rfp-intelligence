
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent

load_dotenv(ROOT / ".env")

DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", ROOT / "outputs"))
LOG_DIR = Path(os.getenv("LOG_DIR", ROOT / "logs"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", ROOT / ".chroma"))

# LLM provider configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "").strip().lower() or None

LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip() or None

ANTHROPIC_API_KEY = (
    os.getenv("ANTHROPIC_API_KEY", "").strip() or None
)

OPENAI_API_KEY = (
    os.getenv("OPENAI_API_KEY", "").strip() or None
)

GEMINI_API_KEY = (
    os.getenv("GEMINI_API_KEY", "").strip() or None
)

LLM_MODEL = os.getenv("LLM_MODEL", "").strip() or None

LLM_BASE_URL = os.getenv(
    "LLM_BASE_URL",
    "https://api.openai.com/v1"
).rstrip("/")

# Embedding and reranking models
EMBED_MODEL = os.getenv(
    "EMBED_MODEL",
    "BAAI/bge-small-en-v1.5"
)

RERANK_MODEL = os.getenv(
    "RERANK_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# Create required directories
for directory in (OUTPUT_DIR, LOG_DIR, CHROMA_DIR):
    directory.mkdir(parents=True, exist_ok=True)
