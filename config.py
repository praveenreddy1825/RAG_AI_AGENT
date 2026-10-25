import os
from pathlib import Path
from dotenv import load_dotenv

# Loading environment
load_dotenv()

class Settings:
    
    # Project Paths
    BASE_DIR: Path = Path(__file__).resolve().parent
    UPLOADS_DIR: Path = BASE_DIR / "uploads"
    
    # Embedding Settings
    # Provider can be 'fastembed' (100% Free, runs locally, no API key required) or 'openai'
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "fastembed").lower()
    FASTEMBED_MODEL: str = os.getenv("FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_LLM_MODEL: str = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
    OPENAI_EMBED_MODEL: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "384" if os.getenv("EMBEDDING_PROVIDER", "fastembed").lower() == "fastembed" else "3072"))

    
    # LLM Provider & Multi-Model Settings
    # Provider can be 'groq' (Free tier, ultra-fast) or 'openai'
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq").lower()
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_GENERATION_MODEL: str = os.getenv("GROQ_GENERATION_MODEL", "openai/gpt-oss-120b")
    GROQ_EVALUATION_MODEL: str = os.getenv("GROQ_EVALUATION_MODEL", "qwen/qwen3.8-27b")

    
    # Qdrant Vector Store Settings
    # Mode can be 'auto', 'local', or 'remote'
    QDRANT_MODE: str = os.getenv("QDRANT_MODE", "auto").lower()
    QDRANT_STORAGE_PATH: str = os.getenv("QDRANT_STORAGE_PATH", str(BASE_DIR / "qdrant_storage"))
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "rag_documents")
    
    # Document Chunking Settings
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "5"))
    
    # Inngest Workflow Settings
    INNGEST_APP_NAME: str = os.getenv("INNGEST_APP_NAME", "rag_agent_app")
    INNGEST_API_BASE: str = os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")

    # LangGraph Agentic RAG Settings
    AGENTIC_RAG_MAX_LOOPS: int = int(os.getenv("AGENTIC_RAG_MAX_LOOPS", "1"))

settings = Settings()
