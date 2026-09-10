import os
import uuid
import logging
import datetime
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, status
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import inngest
import inngest.fast_api
from inngest.experimental import ai
#.\.venv\Scripts\streamlit.exe run streamlit_app.py

from config import settings
from custom_types import (
    RAGChunkAndSrc,
    RAGUpsertResult,
    RAGSearchResult,
    RAQQueryResult,
    QueryRequest,
    CollectionStats,
    AgenticQueryResult,
    AgentStep,
)
from rag_graph import run_agentic_rag
from data_loader import (
    load_and_chunk_document,
    load_and_chunk_pdf,
    embed_texts,
    get_openai_client,
)
from vector_db import QdrantStorage

# Load Environment
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("rag_agent.app")

app = FastAPI(
    title="Production RAG Agent API",
    description="Enterprise-grade Event-Driven Retrieval-Augmented Generation Backend with Inngest & Qdrant",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Inngest Client
inngest_client = inngest.Inngest(
    app_id=settings.INNGEST_APP_NAME,
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)
# Inngest Background Workflows

@inngest_client.create_function(
    fn_id="RAG: Ingest Document",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
    throttle=inngest.Throttle(limit=5, period=datetime.timedelta(minutes=1)),
    rate_limit=inngest.RateLimit(
        limit=5,
        period=datetime.timedelta(hours=1),
        key="event.data.source_id",
    ),
)
async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        file_path = ctx.event.data.get("pdf_path") or ctx.event.data.get("file_path")
        source_id = ctx.event.data.get("source_id", Path(file_path).name)
        doc_chunks = load_and_chunk_document(file_path, source_id=source_id)
        chunks_text = [c.text for c in doc_chunks]
        metadata_list = [c.metadata for c in doc_chunks]
        file_type = doc_chunks[0].file_type if doc_chunks else "unknown"
        return RAGChunkAndSrc(
            chunks=chunks_text,
            source_id=source_id,
            file_type=file_type,
            metadata_list=metadata_list,
        )

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        if not chunks:
            return RAGUpsertResult(ingested=0, source_id=source_id)

        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        metadata = chunks_and_src.metadata_list or [{} for _ in range(len(chunks))]
        
        payloads = [
            {
                "source": source_id,
                "text": chunks[i],
                "chunk_index": i,
                "file_type": chunks_and_src.file_type,
                **(metadata[i] if i < len(metadata) else {}),
            }
            for i in range(len(chunks))
        ]
        
        storage = QdrantStorage()
        storage.upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(chunks), source_id=source_id, collection=storage.collection)

    # Load and chunk document
    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    
    #Embed and upsert into Qdrant
    ingested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)
    
    return ingested.model_dump()


@inngest_client.create_function(
    fn_id="RAG: Query AI",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai"),
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        query_vec = embed_texts([question])[0]
        store = QdrantStorage()
        found = store.search(query_vec, top_k=top_k)
        return RAGSearchResult(
            contexts=found["contexts"],
            sources=found["sources"],
            citations=found["citations"],
        )

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", settings.DEFAULT_TOP_K))
    mode = ctx.event.data.get("mode", "standard").lower()

    #Route to LangGraph Agentic workflow
    if mode == "agentic":
        logger.info(f"Executing LangGraph Agentic RAG workflow for: '{question}'")
        agentic_result = await ctx.step.run(
            "langgraph-agentic-rag",
            lambda: run_agentic_rag(question, top_k=top_k),
        )
        return agentic_result

    #Embed query and retrieve nearest chunks from Qdrant
    found = await ctx.step.run("embed-and-search", lambda: _search(question, top_k), output_type=RAGSearchResult)

    if not found.contexts:
        return {
            "answer": "I could not find any relevant information in the uploaded documents to answer your question.",
            "sources": [],
            "num_contexts": 0,
            "citations": [],
        }

    context_block = "\n\n".join(f"[Doc {i+1}]: {c}" for i, c in enumerate(found.contexts))
    user_content = (
        "You are an expert, precise assistant. Use ONLY the provided context to answer the user question.\n"
        "If the context does not contain enough information to answer, state that clearly.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Provide a clear, detailed, and accurate answer based on the context:"
    )

    adapter = ai.openai.Adapter(
        auth_key=settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY"),
        model=settings.OPENAI_LLM_MODEL,
    )

    #Durable LLM generation via Inngest AI step
    res = await ctx.step.ai.infer(
        "llm-answer",
        adapter=adapter,
        body={
            "max_tokens": 1200,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": "You are a professional RAG assistant that answers questions using only provided context."},
                {"role": "user", "content": user_content},
            ],
        },
    )

    answer = res["choices"][0]["message"]["content"].strip()
    return {
        "answer": answer,
        "sources": found.sources,
        "num_contexts": len(found.contexts),
        "citations": [c.model_dump() if hasattr(c, "model_dump") else c for c in found.citations],
    }


#Serve Inngest Functions via FastAPI
inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])
# Synchronous & REST Endpoints

@app.get("/", tags=["System"])
def root():
    return {
        "name": "Production RAG Agent API",
        "status": "online",
        "docs_url": "/docs",
        "inngest_url": "/api/inngest",
        "version": "1.0.0",
    }


@app.get("/health", tags=["System"])
def health():
    stats = QdrantStorage().get_stats()
    return {
        "status": "healthy",
        "vector_db": stats.model_dump(),
        "embedding_model": settings.OPENAI_EMBED_MODEL,
        "llm_model": settings.OPENAI_LLM_MODEL,
    }

@app.get("/api/v1/stats", response_model=CollectionStats, tags=["Vector Store"])
def get_stats():
    """Returns vector database collection stats and indexed sources."""
    return QdrantStorage().get_stats()


@app.delete("/api/v1/documents/{source_id}", tags=["Vector Store"])
def delete_document(source_id: str):
    """Deletes all chunks for a specific document source."""
    success = QdrantStorage().delete_source(source_id)
    if not success:
        raise HTTPException(status_code=400, detail=f"Failed to delete document '{source_id}'")
    return {"message": f"Successfully deleted '{source_id}'"}


@app.post("/api/v1/query", response_model=RAQQueryResult, tags=["RAG"])
def query_rag_direct(req: QueryRequest):
    """
    Direct synchronous RAG query endpoint.
    Supports both standard single-pass and agentic mode via `mode` field.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    #If agentic mode requested, delegate to LangGraph
    if req.mode.lower() == "agentic":
        res = run_agentic_rag(req.question.strip(), top_k=req.top_k)
        return RAQQueryResult(
            answer=res["answer"],
            sources=res["sources"],
            num_contexts=res["num_contexts"],
            citations=res["citations"],
        )

    store = QdrantStorage()
    query_vec = embed_texts([req.question.strip()])[0]
    search_res = store.search(query_vec, top_k=req.top_k)

    contexts = search_res["contexts"]
    sources = search_res["sources"]
    citations = search_res["citations"]

    if not contexts:
        return RAQQueryResult(
            answer="No relevant documents found to answer your question.",
            sources=[],
            num_contexts=0,
            citations=[],
        )

    context_block = "\n\n".join(f"[Doc {i+1}]: {c}" for i, c in enumerate(contexts))
    prompt = (
        "You are an expert assistant. Answer the question using ONLY the provided context.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {req.question}\n\n"
        "Answer:"
    )

    client = get_openai_client()
    completion = client.chat.completions.create(
        model=settings.OPENAI_LLM_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": "You are a professional RAG assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    answer = completion.choices[0].message.content.strip()

    return RAQQueryResult(
        answer=answer,
        sources=sources,
        num_contexts=len(contexts),
        citations=citations,
    )


@app.post("/api/v1/query/agentic", response_model=AgenticQueryResult, tags=["RAG"])
def query_rag_agentic(req: QueryRequest):
    """
    Agentic Corrective RAG (CRAG) endpoint powered by LangGraph.
    Performs multi-step document grading, query rewriting, and groundedness verification.
    Returns full decision audit trail in `steps`.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = run_agentic_rag(req.question.strip(), top_k=req.top_k)
    return AgenticQueryResult(**result)


@app.post("/api/v1/upload", tags=["Document Ingestion"])
async def upload_and_ingest(file: UploadFile = File(...)):
    """Uploads and ingests a document directly into the vector database."""
    settings.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    target_path = settings.UPLOADS_DIR / file.filename

    with open(target_path, "wb") as f:
        content = await file.read()
        f.write(content)

    doc_chunks = load_and_chunk_document(str(target_path), source_id=file.filename)
    if not doc_chunks:
        raise HTTPException(status_code=400, detail="Could not extract text from document.")

    chunks_text = [c.text for c in doc_chunks]
    vecs = embed_texts(chunks_text)
    ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{file.filename}:{i}")) for i in range(len(chunks_text))]
    payloads = [
        {
            "source": file.filename,
            "text": doc_chunks[i].text,
            "chunk_index": i,
            "file_type": doc_chunks[i].file_type,
            **doc_chunks[i].metadata,
        }
        for i in range(len(doc_chunks))
    ]

    store = QdrantStorage()
    store.upsert(ids, vecs, payloads)

    return {
        "status": "success",
        "file_name": file.filename,
        "chunks_indexed": len(doc_chunks),
        "collection": store.collection,
    }