import os
from pathlib import Path
from dotenv import load_dotenv

# Loading environment from local .env if present
load_dotenv()

_secrets_cache = None


def _get_val(key: str, default: str = "") -> str:
    """
    Retrieves configuration values with priority:
    1. OS environment variables (including .env)
    2. Streamlit Cloud Secrets (st.secrets)
    3. Provided default value
    """
    global _secrets_cache
    val = os.getenv(key)
    if val is not None and val != "":
        return val

    # Cached check for Streamlit Cloud Secrets
    if _secrets_cache is None:
        _secrets_cache = {}
        try:
            import streamlit as st
            if hasattr(st, "secrets"):
                _secrets_cache = dict(st.secrets)
        except Exception:
            _secrets_cache = {}

    return str(_secrets_cache.get(key, default))


class Settings:
    # Project Paths
    BASE_DIR: Path = Path(__file__).resolve().parent
    UPLOADS_DIR: Path = BASE_DIR / "uploads"
    
    # Embedding Settings
    # Provider can be 'fastembed' (100% Free, runs locally, no API key required) or 'openai'
    EMBEDDING_PROVIDER: str = _get_val("EMBEDDING_PROVIDER", "fastembed").lower()
    FASTEMBED_MODEL: str = _get_val("FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5")
    OPENAI_API_KEY: str = _get_val("OPENAI_API_KEY", "")
    OPENAI_LLM_MODEL: str = _get_val("OPENAI_LLM_MODEL", "gpt-4o-mini")
    OPENAI_EMBED_MODEL: str = _get_val("OPENAI_EMBED_MODEL", "text-embedding-3-large")
    EMBEDDING_DIM: int = int(_get_val("EMBEDDING_DIM", "384" if _get_val("EMBEDDING_PROVIDER", "fastembed").lower() == "fastembed" else "3072"))

    # LLM Provider & Multi-Model Settings
    # Provider can be 'groq' (Free tier, ultra-fast) or 'openai'
    LLM_PROVIDER: str = _get_val("LLM_PROVIDER", "groq").lower()
    GROQ_API_KEY: str = _get_val("GROQ_API_KEY", "")
    GROQ_GENERATION_MODEL: str = _get_val("GROQ_GENERATION_MODEL", "openai/gpt-oss-120b")
    GROQ_EVALUATION_MODEL: str = _get_val("GROQ_EVALUATION_MODEL", "qwen/qwen3.8-27b")

    # Qdrant Vector Store Settings
    # Mode can be 'auto', 'local', or 'remote'
    QDRANT_MODE: str = _get_val("QDRANT_MODE", "auto").lower()
    QDRANT_STORAGE_PATH: str = _get_val("QDRANT_STORAGE_PATH", str(BASE_DIR / "qdrant_storage"))
    QDRANT_URL: str = _get_val("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY: str = _get_val("QDRANT_API_KEY", "")
    QDRANT_COLLECTION_NAME: str = _get_val("QDRANT_COLLECTION_NAME", "rag_documents")
    
    # Document Chunking Settings
    CHUNK_SIZE: int = int(_get_val("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(_get_val("CHUNK_OVERLAP", "200"))
    DEFAULT_TOP_K: int = int(_get_val("DEFAULT_TOP_K", "5"))
    
    # Inngest Workflow Settings
    INNGEST_APP_NAME: str = _get_val("INNGEST_APP_NAME", "rag_agent_app")
    INNGEST_API_BASE: str = _get_val("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")

    # LangGraph Agentic RAG Settings
    AGENTIC_RAG_MAX_LOOPS: int = int(_get_val("AGENTIC_RAG_MAX_LOOPS", "1"))


settings = Settings()
