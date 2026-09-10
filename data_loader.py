import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from openai import OpenAI
from config import settings
from custom_types import DocumentChunk

logger = logging.getLogger("rag_agent.data_loader")

# Initialize OpenAI Client
_client: OpenAI = None

def get_openai_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY is not set. OpenAI API calls will fail.")
        _client = OpenAI(api_key=api_key)
    return _client


def _get_splitter():
    try:
        from llama_index.core.node_parser import SentenceSplitter
        return SentenceSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
    except Exception:
        # Fallback simple text splitter if llama-index is not installed
        class SimpleSplitter:
            def __init__(self, chunk_size=1000, chunk_overlap=200):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap

            def split_text(self, text: str) -> List[str]:
                if not text:
                    return []
                chunks = []
                start = 0
                step = self.chunk_size - self.chunk_overlap
                while start < len(text):
                    chunks.append(text[start : start + self.chunk_size])
                    start += step
                return chunks
        return SimpleSplitter(settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)


def extract_text_from_pdf(path: Path) -> List[str]:
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        pages = []
        for p in reader.pages:
            t = p.extract_text()
            if t and t.strip():
                pages.append(t.strip())
        if pages:
            return pages
    except ImportError:
        pass

    # Try llama_index PDFReader
    try:
        from llama_index.readers.file import PDFReader
        docs = PDFReader().load_data(file=str(path))
        return [d.text for d in docs if getattr(d, "text", None)]
    except Exception as e:
        logger.error(f"Error reading PDF {path}: {e}")
        return []


def extract_text_from_docx(path: Path) -> List[str]:
    try:
        import docx
        doc = docx.Document(str(path))
        full_text = []
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text.strip())
        return ["\n\n".join(full_text)] if full_text else []
    except ImportError:
        logger.warning("python-docx not installed; attempting raw text fallback.")
        return []
    except Exception as e:
        logger.error(f"Error reading Word document {path}: {e}")
        return []


def extract_text_from_plain(path: Path) -> List[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        return [text] if text.strip() else []
    except Exception as e:
        logger.error(f"Error reading text file {path}: {e}")
        return []


def load_and_chunk_document(file_path: str, source_id: str = None) -> List[DocumentChunk]:      

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()
    src = source_id or path.name

    if ext == ".pdf":
        sections = extract_text_from_pdf(path)
        file_type = "pdf"
    elif ext in (".docx", ".doc"):
        sections = extract_text_from_docx(path)
        file_type = "docx"
    elif ext in (".txt", ".md", ".csv", ".json", ".log"):
        sections = extract_text_from_plain(path)
        file_type = ext.lstrip(".")
    else:
        # Default fallback to plain text reader
        sections = extract_text_from_plain(path)
        file_type = "unknown"

    splitter = _get_splitter()
    chunks: List[DocumentChunk] = []
    chunk_idx = 0

    for section_idx, section in enumerate(sections):
        split_texts = splitter.split_text(section)
        for text in split_texts:
            text = text.strip()
            if text:
                chunks.append(
                    DocumentChunk(
                        text=text,
                        source_id=src,
                        chunk_index=chunk_idx,
                        file_type=file_type,
                        metadata={
                            "section_index": section_idx,
                            "char_count": len(text),
                            "file_name": path.name,
                        },
                    )
                )
                chunk_idx += 1

    logger.info(f"Loaded '{path.name}' ({file_type}): generated {len(chunks)} chunks.")
    return chunks


def load_and_chunk_pdf(path: str) -> List[str]:
    doc_chunks = load_and_chunk_document(path)
    return [c.text for c in doc_chunks]


def embed_texts(texts: List[str], batch_size: int = 64) -> List[List[float]]:
    if not texts:
        return []

    client = get_openai_client()
    embeddings: List[List[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(
            model=settings.OPENAI_EMBED_MODEL,
            input=batch,
        )
        batch_vecs = [item.embedding for item in response.data]
        embeddings.extend(batch_vecs)

    return embeddings