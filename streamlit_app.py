import asyncio
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import streamlit as st
import requests
from dotenv import load_dotenv

import inngest
from config import settings
from custom_types import DocumentChunk
from data_loader import load_and_chunk_document, embed_texts
from vector_db import QdrantStorage
from rag_graph import run_agentic_rag

# Load environment
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG AI Agent Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown(
    """
    <style>
    /* Modern typography and card styling */
    .stApp {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header card */
    .hero-card {
        padding: 1.5rem;
        border-radius: 12px;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    
    .hero-title {
        font-size: 2.1rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
    }
    
    /* Status chips */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-right: 8px;
    }
    .status-badge.online {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .status-badge.local {
        background-color: rgba(59, 130, 246, 0.2);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.4);
    }
    
    /* Citation container */
    .citation-card {
        padding: 0.75rem 1rem;
        border-radius: 8px;
        background: rgba(30, 41, 59, 0.5);
        border-left: 3px solid #818cf8;
        margin: 0.5rem 0;
        font-size: 0.88rem;
    }

    /* Agent Reasoning Trace styling */
    .agent-trace-step {
        padding: 8px 12px;
        margin: 6px 0;
        border-radius: 6px;
        background: rgba(15, 23, 42, 0.6);
        border-left: 3px solid #38bdf8;
        font-size: 0.86rem;
    }
    .agent-node-badge {
        font-family: monospace;
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 600;
        margin-right: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Inngest Client
@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(app_id=settings.INNGEST_APP_NAME, is_production=False)


def get_vector_store() -> QdrantStorage:
    return QdrantStorage()


def save_uploaded_file(file) -> Path:
    settings.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    file_path = settings.UPLOADS_DIR / file.name
    file_path.write_bytes(file.getbuffer())
    return file_path


async def send_inngest_ingest_event(file_path: Path) -> None:
    client = get_inngest_client()
    await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_path": str(file_path.resolve()),
                "file_path": str(file_path.resolve()),
                "source_id": file_path.name,
            },
        )
    )


async def send_inngest_query_event(question: str, top_k: int, mode: str = "standard") -> str:
    client = get_inngest_client()
    result = await client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={
                "question": question,
                "top_k": top_k,
                "mode": mode,
            },
        )
    )
    return result[0]


def fetch_inngest_runs(event_id: str) -> list[dict]:
    url = f"{settings.INNGEST_API_BASE}/events/{event_id}/runs"
    resp = requests.get(url, timeout=5)
    resp.raise_for_status()
    return resp.json().get("data", [])


def wait_for_inngest_output(event_id: str, timeout_s: float = 60.0, poll_interval_s: float = 0.5) -> dict:
    start = time.time()
    last_status = None
    while True:
        try:
            runs = fetch_inngest_runs(event_id)
            if runs:
                run = runs[0]
                status = run.get("status")
                last_status = status or last_status
                if status in ("Completed", "Succeeded", "Success", "Finished"):
                    return run.get("output") or {}
                if status in ("Failed", "Cancelled"):
                    raise RuntimeError(f"Workflow execution {status}")
        except Exception as e:
            if "Workflow execution" in str(e):
                raise
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Workflow timed out (status: {last_status})")
        time.sleep(poll_interval_s)


def direct_rag_query(question: str, top_k: int) -> dict:
    """Direct query fallback using local Qdrant and OpenAI without needing Inngest Dev Server."""
    store = get_vector_store()
    query_vec = embed_texts([question])[0]
    search_res = store.search(query_vec, top_k=top_k)

    contexts = search_res["contexts"]
    sources = search_res["sources"]
    citations = search_res["citations"]

    if not contexts:
        return {
            "answer": "No relevant document chunks found for your query. Please upload documents first.",
            "sources": [],
            "citations": [],
            "num_contexts": 0,
        }

    context_block = "\n\n".join(f"[Doc {i+1}]: {c}" for i, c in enumerate(contexts))
    prompt = (
        "You are an expert AI assistant. Answer the user's question using ONLY the provided context.\n"
        "Cite the sources or document details if available.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Clear and concise answer:"
    )

    from data_loader import get_openai_client
    client = get_openai_client()
    completion = client.chat.completions.create(
        model=settings.OPENAI_LLM_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": "You are a precise, grounded RAG assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    answer = completion.choices[0].message.content.strip()

    return {
        "answer": answer,
        "sources": sources,
        "citations": citations,
        "num_contexts": len(contexts),
    }


def direct_document_ingest(file_path: Path) -> int:
    """Directly chunks and vectors a document into Qdrant."""
    import uuid
    doc_chunks = load_and_chunk_document(str(file_path), source_id=file_path.name)
    if not doc_chunks:
        return 0

    chunks_text = [c.text for c in doc_chunks]
    vecs = embed_texts(chunks_text)
    ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{file_path.name}:{i}")) for i in range(len(chunks_text))]
    payloads = [
        {
            "source": file_path.name,
            "text": doc_chunks[i].text,
            "chunk_index": i,
            "file_type": doc_chunks[i].file_type,
            **doc_chunks[i].metadata,
        }
        for i in range(len(doc_chunks))
    ]
    store = get_vector_store()
    store.upsert(ids, vecs, payloads)
    return len(doc_chunks)

with st.sidebar:
    st.markdown("### ⚡ System Dashboard")
    
    # Check Vector Store Stats
    stats = get_vector_store().get_stats()
    mode_label = "Local Disk" if stats.mode == "local" else "Remote Server"
    
    st.markdown(
        f"""
        <span class="status-badge online">● System Active</span>
        <span class="status-badge local">📦 {mode_label}</span>
        """,
        unsafe_allow_html=True,
    )
    
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Vectors", stats.total_vectors)
    with col2:
        st.metric("Indexed Docs", len(stats.sources))

    st.markdown("#### 📁 Indexed Documents")
    if stats.sources:
        for src in stats.sources:
            c1, c2 = st.columns([4, 1])
            with c1:
                st.caption(f"📄 {src}")
            with c2:
                if st.button("✕", key=f"del_{src}", help=f"Delete {src}"):
                    get_vector_store().delete_source(src)
                    st.toast(f"Deleted {src}")
                    st.rerun()
    else:
        st.info("No documents uploaded yet.")

    st.divider()
    st.markdown("#### ⚙️ Retrieval Settings")
    
    rag_mode = st.radio(
        "🤖 RAG Engine Mode",
        ["🧠 LangGraph Agentic RAG", "⚡ Standard RAG"],
        index=0,
        help="LangGraph Agentic RAG performs document grading, query rewriting, and groundedness verification with a full step audit trace."
    )
    is_agentic = "Agentic" in rag_mode

    top_k = st.slider("Context Chunks (Top-K)", min_value=1, max_value=15, value=settings.DEFAULT_TOP_K)
    
    use_inngest = st.toggle(
        "Use Inngest Dev Server",
        value=False,
        help="Route workflows through Inngest event bus (requires 'npx inngest-cli@latest dev' running)",
    )

    st.caption(f"**LLM:** `{settings.OPENAI_LLM_MODEL}`")
    st.caption(f"**Embedder:** `{settings.OPENAI_EMBED_MODEL}`")

st.markdown(
    f"""
    <div class="hero-card">
        <div class="hero-title">RAG AI Agent Studio</div>
        <div class="hero-subtitle">
            Enterprise Retrieval-Augmented Generation powered by Inngest Durable Orchestration & Qdrant Vector Storage.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
tab_chat, tab_upload, tab_inspector = st.tabs(["💬 Ask & Chat", "📤 Ingest Documents", "🔍 Document Inspector"])
with tab_chat:
    # Initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Render Agent Reasoning Trace if present
            if "steps" in message and message["steps"]:
                with st.expander(f"🧠 Agent Reasoning & Reflection Trace ({len(message['steps'])} steps)"):
                    for s in message["steps"]:
                        node = s.get("node", "step").upper()
                        desc = s.get("description", "")
                        status = s.get("status", "completed")
                        icon = "✅" if status == "completed" else ("🔄" if node == "REWRITE_QUERY" else "⚠️")
                        st.markdown(
                            f"<div class='agent-trace-step'>"
                            f"<span>{icon}</span> <span class='agent-node-badge'>{node}</span>"
                            f"<span>{desc}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                        if s.get("details"):
                            det = s["details"]
                            if "rewritten_query" in det:
                                st.caption(f"↳ **Rewritten Query:** `{det['rewritten_query']}`")

            if "rewritten_query" in message and message["rewritten_query"]:
                st.info(f"💡 **Search Query Refined by Agent:** *\"{message['rewritten_query']}\"*")

            if "citations" in message and message["citations"]:
                with st.expander(f"📚 Sources & Citations ({len(message['citations'])})"):
                    for idx, cit in enumerate(message["citations"]):
                        src = cit.get("source", "Document")
                        score = cit.get("score")
                        score_text = f" • Match: {score*100:.1f}%" if score else ""
                        st.markdown(
                            f"**[{idx+1}] {src}**{score_text}\n"
                            f"> {cit.get('text', '')}\n"
                        )

    # Chat Input
    prompt_input = st.chat_input("Ask a question about your documents...")

    if prompt_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt_input})
        with st.chat_message("user"):
            st.markdown(prompt_input)

        # Generate assistant response
        with st.chat_message("assistant"):
            spinner_msg = (
                "🧠 LangGraph Agent reasoning: retrieving, grading relevance, and checking facts..."
                if is_agentic
                else "Searching document context and synthesizing answer..."
            )
            with st.spinner(spinner_msg):
                try:
                    if is_agentic:
                        # LangGraph Agentic RAG Execution
                        if use_inngest:
                            event_id = asyncio.run(send_inngest_query_event(prompt_input, top_k, mode="agentic"))
                            output = wait_for_inngest_output(event_id)
                        else:
                            output = run_agentic_rag(prompt_input, top_k=top_k)

                        answer = output.get("answer", "")
                        citations = output.get("citations", [])
                        steps = output.get("steps", [])
                        rewritten_query = output.get("rewritten_query")
                    else:
                        # Standard RAG Execution
                        if use_inngest:
                            event_id = asyncio.run(send_inngest_query_event(prompt_input, top_k, mode="standard"))
                            output = wait_for_inngest_output(event_id)
                        else:
                            output = direct_rag_query(prompt_input, top_k)

                        answer = output.get("answer", "")
                        citations = output.get("citations", [])
                        steps = []
                        rewritten_query = None

                    st.markdown(answer)

                    # Show Reasoning Trace
                    if steps:
                        with st.expander(f"🧠 Agent Reasoning & Reflection Trace ({len(steps)} steps)", expanded=True):
                            for s in steps:
                                node = s.get("node", "step").upper()
                                desc = s.get("description", "")
                                status = s.get("status", "completed")
                                icon = "✅" if status == "completed" else ("🔄" if node == "REWRITE_QUERY" else "⚠️")
                                st.markdown(
                                    f"<div class='agent-trace-step'>"
                                    f"<span>{icon}</span> <span class='agent-node-badge'>{node}</span>"
                                    f"<span>{desc}</span>"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )
                                if s.get("details"):
                                    det = s["details"]
                                    if "rewritten_query" in det:
                                        st.caption(f"↳ **Rewritten Query:** `{det['rewritten_query']}`")

                    if rewritten_query:
                        st.info(f"💡 **Search Query Refined by Agent:** *\"{rewritten_query}\"*")

                    if citations:
                        with st.expander(f"📚 Sources & Citations ({len(citations)})"):
                            for idx, cit in enumerate(citations):
                                src = cit.get("source", "Document")
                                score = cit.get("score")
                                score_text = f" • Match: {score*100:.1f}%" if score else ""
                                st.markdown(
                                    f"**[{idx+1}] {src}**{score_text}\n"
                                    f"> {cit.get('text', '')}\n"
                                )

                    # Record to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "citations": citations,
                        "steps": steps,
                        "rewritten_query": rewritten_query,
                    })
                except Exception as e:
                    error_msg = f"⚠️ Query failed: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
with tab_upload:
    st.subheader("Upload Documents to Ingest")
    st.caption("Supports **PDF** (`.pdf`), **Word** (`.docx`), **Markdown** (`.md`), and **Plain Text** (`.txt`).")

    uploaded_files = st.file_uploader(
        "Choose documents",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button("🚀 Process & Index Documents", type="primary"):
            progress_bar = st.progress(0.0)
            total = len(uploaded_files)

            for i, up_file in enumerate(uploaded_files):
                with st.status(f"Ingesting '{up_file.name}'...", expanded=True) as status:
                    saved_path = save_uploaded_file(up_file)
                    st.write("Saved file to local disk.")

                    if use_inngest:
                        st.write("Triggering Inngest durable event...")
                        asyncio.run(send_inngest_ingest_event(saved_path))
                        status.update(label=f"Ingestion event sent for '{up_file.name}'!", state="complete")
                    else:
                        st.write("Parsing, splitting sentences, and embedding vectors...")
                        count = direct_document_ingest(saved_path)
                        status.update(label=f"Successfully indexed '{up_file.name}' ({count} chunks)!", state="complete")

                    progress_bar.progress((i + 1) / total)

            st.success(f"Finished processing {total} document(s)!")
            time.sleep(1)
            st.rerun()
with tab_inspector:
    st.subheader("Inspect Indexed Vectors & Payloads")
    sources = get_vector_store().list_sources()

    if not sources:
        st.info("No indexed documents found in Qdrant collection.")
    else:
        selected_source = st.selectbox("Select document to inspect:", sources)
        if selected_source:
            # Query sample chunks for this source
            try:
                store = get_vector_store()
                scroll_res, _ = store.client.scroll(
                    collection_name=store.collection,
                    limit=20,
                    with_payload=True,
                    with_vectors=False,
                )
                matching = [p for p in scroll_res if getattr(p, "payload", {}).get("source") == selected_source]
                
                st.write(f"Displaying **{len(matching)}** sample chunk(s) for `{selected_source}`:")
                for p in matching:
                    payload = getattr(p, "payload", {})
                    idx = payload.get("chunk_index", 0)
                    with st.expander(f"Chunk #{idx} (ID: {p.id})"):
                        st.text(payload.get("text", ""))
                        st.json({k: v for k, v in payload.items() if k != "text"})
            except Exception as e:
                st.error(f"Error inspecting vectors: {e}")
