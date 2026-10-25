import os
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

# Loading environment from local .env if present
load_dotenv()


def _clean_val(val: Any) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    # Strip any accidental surrounding quotes (e.g. if pasted as '"gsk_..."' or "'gsk_...'")
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    return s


def _get_val(key: str, default: str = "") -> str:
    """
    Retrieves configuration values dynamically with priority:
    1. OS environment variables (including .env)
    2. Streamlit Cloud Secrets (st.secrets), supporting case-insensitivity & sections
    3. Provided default value
    """
    # 1. OS environment variable
    val = os.getenv(key)
    if val is not None and val.strip() != "":
        return _clean_val(val)

    # 2. Live lookup in Streamlit Cloud Secrets
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            # Direct match
            if key in st.secrets:
                return _clean_val(st.secrets[key])
            
            # Case-insensitive top-level match
            key_lower = key.lower()
            for sec_k, sec_v in st.secrets.items():
                if sec_k.lower() == key_lower:
                    if not isinstance(sec_v, dict):
                        return _clean_val(sec_v)
                elif isinstance(sec_v, dict):
                    # Check nested table, e.g. [groq] api_key = "..."
                    for sub_k, sub_v in sec_v.items():
                        if sub_k.lower() == key_lower or sub_k.lower() == key_lower.replace("groq_", "").replace("qdrant_", ""):
                            return _clean_val(sub_v)
    except Exception:
        pass

    return default


class Settings:
    # Project Paths
    BASE_DIR: Path = Path(__file__).resolve().parent
    UPLOADS_DIR: Path = BASE_DIR / "uploads"

    @property
    def EMBEDDING_PROVIDER(self) -> str:
        return _get_val("EMBEDDING_PROVIDER", "fastembed").lower()

    @property
    def FASTEMBED_MODEL(self) -> str:
        return _get_val("FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5")

    @property
    def OPENAI_API_KEY(self) -> str:
        return _get_val("OPENAI_API_KEY", "")

    @property
    def OPENAI_LLM_MODEL(self) -> str:
        return _get_val("OPENAI_LLM_MODEL", "gpt-4o-mini")

    @property
    def OPENAI_EMBED_MODEL(self) -> str:
        return _get_val("OPENAI_EMBED_MODEL", "text-embedding-3-large")

    @property
    def EMBEDDING_DIM(self) -> int:
        return int(_get_val("EMBEDDING_DIM", "384" if self.EMBEDDING_PROVIDER == "fastembed" else "3072"))

    @property
    def LLM_PROVIDER(self) -> str:
        return _get_val("LLM_PROVIDER", "groq").lower()

    @property
    def GROQ_API_KEY(self) -> str:
        return _get_val("GROQ_API_KEY", "")

    @property
    def GROQ_GENERATION_MODEL(self) -> str:
        return _get_val("GROQ_GENERATION_MODEL", "openai/gpt-oss-120b")

    @property
    def GROQ_EVALUATION_MODEL(self) -> str:
        return _get_val("GROQ_EVALUATION_MODEL", "qwen/qwen3.8-27b")

    @property
    def QDRANT_MODE(self) -> str:
        return _get_val("QDRANT_MODE", "auto").lower()

    @property
    def QDRANT_STORAGE_PATH(self) -> str:
        return _get_val("QDRANT_STORAGE_PATH", str(self.BASE_DIR / "qdrant_storage"))

    @property
    def QDRANT_URL(self) -> str:
        return _get_val("QDRANT_URL", "http://localhost:6333")

    @property
    def QDRANT_API_KEY(self) -> str:
        return _get_val("QDRANT_API_KEY", "")

    @property
    def QDRANT_COLLECTION_NAME(self) -> str:
        return _get_val("QDRANT_COLLECTION_NAME", "rag_documents")

    @property
    def CHUNK_SIZE(self) -> int:
        return int(_get_val("CHUNK_SIZE", "1000"))

    @property
    def CHUNK_OVERLAP(self) -> int:
        return int(_get_val("CHUNK_OVERLAP", "200"))

    @property
    def DEFAULT_TOP_K(self) -> int:
        return int(_get_val("DEFAULT_TOP_K", "5"))

    @property
    def INNGEST_APP_NAME(self) -> str:
        return _get_val("INNGEST_APP_NAME", "rag_agent_app")

    @property
    def INNGEST_API_BASE(self) -> str:
        return _get_val("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")

    @property
    def AGENTIC_RAG_MAX_LOOPS(self) -> int:
        return int(_get_val("AGENTIC_RAG_MAX_LOOPS", "1"))


settings = Settings()
