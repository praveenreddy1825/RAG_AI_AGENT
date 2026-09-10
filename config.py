import os
from pathlib import Path
from dotenv import load_dotenv

# Loading environment
load_dotenv()

class Settings:
    
    # Project Paths
    BASE_DIR: Path = Path(__file__).resolve().parent
    UPLOADS_DIR: Path = BASE_DIR / "uploads"
    
    # OpenAI Settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_LLM_MODEL: str = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
    OPENAI_EMBED_MODEL: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "3072"))
    
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
    INNGEST_DEV_SERVER_URL: str = os.getenv("INNGEST_DEV_SERVER_URL", "http://127.0.0.1:8288")
    INNGEST_API_BASE: str = os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")

    # LangGraph Agentic RAG Settings
    AGENTIC_RAG_ENABLED: bool = os.getenv("AGENTIC_RAG_ENABLED", "true").lower() in ("true", "1", "yes")
    AGENTIC_RAG_MAX_LOOPS: int = int(os.getenv("AGENTIC_RAG_MAX_LOOPS", "2"))

settings = Settings()
