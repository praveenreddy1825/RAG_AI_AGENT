from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """Represents an individual text chunk extracted from a document."""
    text: str
    source_id: str
    chunk_index: int
    file_type: str
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    """Detailed citation reference for retrieved contexts."""
    text: str
    source: str
    score: Optional[float] = None
    chunk_index: Optional[int] = None
    user_id: Optional[str] = None


class RAGChunkAndSrc(BaseModel):
    """Legacy and Inngest step data carrier for document chunks."""
    chunks: List[str]
    source_id: Optional[str] = None
    file_type: Optional[str] = "pdf"
    metadata_list: Optional[List[Dict[str, Any]]] = None


class RAGUpsertResult(BaseModel):
    """Result of vector database upsert operation."""
    ingested: int
    source_id: Optional[str] = None
    collection: Optional[str] = None


class RAGSearchResult(BaseModel):
    """Result from vector search retrieval."""
    contexts: List[str]
    sources: List[str]
    citations: List[Citation] = Field(default_factory=list)


class RAGQueryResult(BaseModel):
    """RAG model answer generation result."""
    answer: str
    sources: List[str]
    num_contexts: int
    citations: List[Citation] = Field(default_factory=list)


# Alias for backward compatibility
RAQQueryResult = RAGQueryResult


class CollectionStats(BaseModel):
    """Vector database collection statistics and health."""
    collection_name: str
    status: str
    mode: str
    total_vectors: int
    sources: List[str] = Field(default_factory=list)


class AgentStep(BaseModel):
    """Represents an individual step or decision executed in the agentic graph."""
    node: str
    status: str
    description: str
    details: Optional[Dict[str, Any]] = None


class AgenticQueryResult(BaseModel):
    """Result from LangGraph Agentic RAG execution."""
    answer: str
    sources: List[str] = Field(default_factory=list)
    num_contexts: int = 0
    citations: List[Citation] = Field(default_factory=list)
    steps: List[AgentStep] = Field(default_factory=list)
    original_question: str
    rewritten_query: Optional[str] = None
    is_grounded: bool = True
    iterations: int = 1
    user_id: Optional[str] = None


class QueryRequest(BaseModel):
    """API request model for querying RAG."""
    question: str
    top_k: int = 5
    mode: str = "standard"  # "standard" or "agentic"
    user_id: Optional[str] = None


CONVERSATIONAL_GREETINGS: set = {
    "hi", "hello", "hey", "hola", "greetings", "good morning", "good afternoon", "good evening",
    "hi there", "hello there", "hey there", "who are you", "what can you do", "help", "howdy",
    "thank you", "thanks", "bye", "goodbye"
}