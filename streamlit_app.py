import uuid
import asyncio
import time
from pathlib import Path
from typing import Optional

import streamlit as st
import requests

import inngest
from langchain_core.messages import SystemMessage, HumanMessage
from config import settings
from custom_types import CONVERSATIONAL_GREETINGS
from data_loader import load_and_chunk_document, embed_texts
from vector_db import QdrantStorage
from rag_graph import run_agentic_rag, get_llm
from auth import render_auth_ui

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

    /* Workflow card & timeline */
    .workflow-timeline-card {
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
        border-radius: 10px;
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-left: 4px solid #38bdf8;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        transition: all 0.2s ease;
    }
    .workflow-timeline-card:hover {
        border-color: rgba(255, 255, 255, 0.2);
        background: rgba(30, 41, 59, 0.65);
    }
    .workflow-node-badge {
        font-family: monospace;
        font-size: 0.8rem;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 700;
        letter-spacing: 0.04em;
        display: inline-block;
        margin-right: 8px;
    }
    .node-retrieve {
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    .node-grade {
        background: rgba(168, 85, 247, 0.15);
        color: #c084fc;
        border: 1px solid rgba(168, 85, 247, 0.3);
    }
    .node-rewrite {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .node-generate {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .node-groundedness {
        background: rgba(99, 102, 241, 0.15);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    .node-error {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .blueprint-card {
        padding: 1.25rem;
        border-radius: 12px;
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 1rem;
    }
    .step-number {
        font-weight: 700;
        color: #94a3b8;
        font-size: 0.8rem;
        text-transform: uppercase;
        margin-bottom: 4px;
        letter-spacing: 0.05em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Inngest Client
@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(app_id=settings.INNGEST_APP_NAME, is_production=False)


@st.cache_resource
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


def direct_rag_query(question: str, top_k: int, user_id: Optional[str] = None) -> dict:
    """Direct query fallback using local Qdrant and OpenAI without needing Inngest Dev Server."""
    clean_q = question.strip().lower().rstrip("!?.")
    store = get_vector_store()
    stats = store.get_stats(user_id=user_id)

    if clean_q in CONVERSATIONAL_GREETINGS:
        if stats.sources:
            doc_str = ", ".join(f"`{s}`" for s in stats.sources)
            greeting_msg = (
                f"Hello! 👋 I am your RAG AI Assistant.\n\n"
                f"I currently have access to your indexed document(s): {doc_str}.\n\n"
                f"What would you like to know or analyze from these documents?"
            )
        else:
            greeting_msg = (
                "Hello! 👋 I am your RAG AI Assistant.\n\n"
                "I don't see any uploaded documents yet. Please upload your documents in the **📤 Document Ingestion** tab, and I'll be ready to answer your questions!"
            )
        return {
            "answer": greeting_msg,
            "sources": stats.sources,
            "citations": [],
            "num_contexts": 0,
        }

    query_vec = embed_texts([question])[0]
    search_res = store.search(query_vec, top_k=top_k, user_id=user_id)

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

    llm = get_llm(role="generator", temperature=0.2)
    response = llm.invoke([
        SystemMessage(content="You are a precise, grounded RAG assistant."),
        HumanMessage(content=prompt),
    ])
    answer = response.content.strip()

    return {
        "answer": answer,
        "sources": sources,
        "citations": citations,
        "num_contexts": len(contexts),
    }


def direct_document_ingest(file_path: Path, user_id: Optional[str] = None) -> int:
    """Directly chunks and vectors a document into Qdrant for a specific user."""
    doc_chunks = load_and_chunk_document(str(file_path), source_id=file_path.name)
    if not doc_chunks:
        return 0

    chunks_text = [c.text for c in doc_chunks]
    vecs = embed_texts(chunks_text)
    user_prefix = f"{user_id}:" if user_id else ""
    ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{user_prefix}{file_path.name}:{i}")) for i in range(len(chunks_text))]
    payloads = [
        {
            "source": file_path.name,
            "text": doc_chunks[i].text,
            "chunk_index": i,
            "file_type": doc_chunks[i].file_type,
            "user_id": user_id,
            **doc_chunks[i].metadata,
        }
        for i in range(len(doc_chunks))
    ]
    store = get_vector_store()
    store.upsert(ids, vecs, payloads, user_id=user_id)
    return len(doc_chunks)


# -------------------------------------------------------------
# Multi-User Authentication Guard
# -------------------------------------------------------------
if "user" not in st.session_state:
    render_auth_ui()
    st.stop()

current_user = st.session_state.get("user")
if not current_user:
    st.stop()
current_user_id = current_user.get("username", "guest")

with st.sidebar:
    st.markdown(
        f"""
        <div style="padding: 12px 14px; background: rgba(30, 41, 59, 0.7); border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.1); margin-bottom: 14px;">
            <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">Current User</div>
            <div style="font-size: 1.05rem; font-weight: 700; color: #38bdf8; margin: 2px 0;">👤 {current_user['display_name']}</div>
            <div style="font-size: 0.8rem; color: #cbd5e1;">@{current_user['username']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("🚪 Sign Out", key="btn_signout", use_container_width=True):
        del st.session_state["user"]
        st.session_state.messages = []
        st.session_state.workflow_history = []
        st.rerun()

    st.markdown("### ⚡ System Dashboard")
    
    # Check Vector Store Stats for current user
    stats = get_vector_store().get_stats(user_id=current_user_id)
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
                    get_vector_store().delete_source(src, user_id=current_user_id)
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

    if settings.LLM_PROVIDER == "groq":
        st.caption(f"**⚡ Provider:** `Groq (Free Tier)`")
        st.caption(f"**🚀 Generator LLM:** `{settings.GROQ_GENERATION_MODEL}`")
        st.caption(f"**🔍 Evaluator LLM:** `{settings.GROQ_EVALUATION_MODEL}`")
    else:
        st.caption(f"**LLM:** `{settings.OPENAI_LLM_MODEL}`")
    if settings.EMBEDDING_PROVIDER == "fastembed":
        st.caption(f"**📦 Embedder:** `FastEmbed (Local / Free)`")
    else:
        st.caption(f"**📦 Embedder:** `{settings.OPENAI_EMBED_MODEL}`")

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
tab_chat, tab_workflow, tab_upload, tab_inspector = st.tabs([
    "💬 Response & Chat",
    "🧠 Agent Workflow",
    "📤 Document Ingestion",
    "🔍 Document Inspector",
])

# Initialize session state for messages and workflow history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "workflow_history" not in st.session_state:
    st.session_state.workflow_history = []

with tab_chat:
    c_head1, c_head2 = st.columns([6, 1])
    with c_head1:
        st.subheader("💬 Ask Your Documents")
        st.caption("Clean conversational interface. Focused purely on answers with verified context citations.")
    with c_head2:
        if st.session_state.messages and st.button("🗑️ Clear", help="Clear conversation history"):
            st.session_state.messages = []
            st.rerun()

    # Display chat history (clean response ONLY)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Compact expandable citations
            citations = message.get("citations", [])
            if citations:
                with st.expander(f"📚 Cited Sources ({len(citations)})", expanded=False):
                    for idx, cit in enumerate(citations):
                        src = cit.get("source", "Document")
                        score = cit.get("score")
                        score_text = f" • Match: {score*100:.1f}%" if score else ""
                        st.markdown(f"**[{idx+1}] {src}**{score_text}")
                        sample_txt = cit.get("text", "").strip()
                        if sample_txt:
                            st.caption(f"> {sample_txt[:280]}{'...' if len(sample_txt) > 280 else ''}")

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
                "🧠 Executing LangGraph CRAG agent: retrieval, relevance grading, and grounded generation..."
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
                            output = run_agentic_rag(prompt_input, top_k=top_k, user_id=current_user_id)

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
                            output = direct_rag_query(prompt_input, top_k, user_id=current_user_id)

                        answer = output.get("answer", "")
                        citations = output.get("citations", [])
                        steps = [
                            {
                                "node": "retrieve",
                                "status": "completed",
                                "description": f"Retrieved {len(citations)} candidate chunk(s) from Qdrant vector database.",
                                "details": {"chunks_found": len(citations)},
                            },
                            {
                                "node": "generate",
                                "status": "completed",
                                "description": "Synthesized direct answer using retrieved chunks.",
                                "details": {},
                            },
                        ]
                        rewritten_query = None

                    # 1. Clean Final Response Display
                    st.markdown(answer)

                    # 2. Compact sources expander
                    if citations:
                        with st.expander(f"📚 Cited Sources ({len(citations)})", expanded=False):
                            for idx, cit in enumerate(citations):
                                src = cit.get("source", "Document")
                                score = cit.get("score")
                                score_text = f" • Match: {score*100:.1f}%" if score else ""
                                st.markdown(f"**[{idx+1}] {src}**{score_text}")
                                sample_txt = cit.get("text", "").strip()
                                if sample_txt:
                                    st.caption(f"> {sample_txt[:280]}{'...' if len(sample_txt) > 280 else ''}")

                    # 3. Notification to inspect workflow in the dedicated tab
                    st.caption("✨ *Execution trace and reasoning graph are available in the **🧠 Agent Workflow** tab.*")

                    # 4. Save clean response to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "citations": citations,
                    })

                    # 5. Record run to workflow history for dedicated inspection
                    st.session_state.workflow_history.append({
                        "id": len(st.session_state.workflow_history) + 1,
                        "timestamp": time.strftime("%H:%M:%S"),
                        "query": prompt_input,
                        "mode": "LangGraph Agentic RAG" if is_agentic else "Standard RAG",
                        "answer": answer,
                        "steps": steps,
                        "citations": citations,
                        "sources": output.get("sources", []),
                        "rewritten_query": rewritten_query,
                        "is_grounded": output.get("is_grounded", True),
                        "iterations": output.get("iterations", 1),
                        "num_contexts": output.get("num_contexts", len(citations)),
                    })
                except Exception as e:
                    error_msg = f"⚠️ Query failed: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

with tab_workflow:
    st.subheader("🧠 Agent Workflow & Reasoning Studio")
    st.caption("Inspect multi-stage retrieval, document grading, conditional query reformulation, and hallucination verification.")

    if not st.session_state.workflow_history:
        st.info("💡 No agent queries executed in this session yet. Run a question in **💬 Response & Chat** to see the live graph execution steps here!")
        
        st.markdown("#### 🗺️ Agentic CRAG Architecture & Lifecycle")
        col_w1, col_w2 = st.columns(2)
        with col_w1:
            st.markdown(
                """
                <div class="blueprint-card">
                    <div class="step-number">Step 01 • Vector DB Retrieval</div>
                    <span class="workflow-node-badge node-retrieve">RETRIEVE</span>
                    <p style="color: #cbd5e1; font-size: 0.9rem; margin-top: 8px;">
                        Fetches candidate chunks from Qdrant vector storage using fast 384-dimensional dense semantic embeddings.
                    </p>
                </div>
                <div class="blueprint-card">
                    <div class="step-number">Step 02 • Relevance Grading & Filtering</div>
                    <span class="workflow-node-badge node-grade">GRADE_DOCUMENTS</span>
                    <p style="color: #cbd5e1; font-size: 0.9rem; margin-top: 8px;">
                        Evaluator LLM strictly checks whether candidate chunks are topically relevant, filtering out misleading or irrelevant text.
                    </p>
                </div>
                <div class="blueprint-card">
                    <div class="step-number">Step 03 • Conditional Query Rewriting</div>
                    <span class="workflow-node-badge node-rewrite">REWRITE_QUERY</span>
                    <p style="color: #cbd5e1; font-size: 0.9rem; margin-top: 8px;">
                        If retrieved chunks fail relevance criteria, the agent automatically reformulates the search query and re-retrieves context.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_w2:
            st.markdown(
                """
                <div class="blueprint-card">
                    <div class="step-number">Step 04 • Context-Grounded Generation</div>
                    <span class="workflow-node-badge node-generate">GENERATE</span>
                    <p style="color: #cbd5e1; font-size: 0.9rem; margin-top: 8px;">
                        Generator LLM synthesizes an authoritative answer using exclusively the verified relevant document chunks.
                    </p>
                </div>
                <div class="blueprint-card">
                    <div class="step-number">Step 05 • Hallucination & Fact Check</div>
                    <span class="workflow-node-badge node-groundedness">CHECK_GROUNDEDNESS</span>
                    <p style="color: #cbd5e1; font-size: 0.9rem; margin-top: 8px;">
                        Evaluator verifies whether each factual statement in the answer is grounded in source text, looping back to rewrite if hallucination occurs.
                    </p>
                </div>
                <div class="blueprint-card">
                    <div class="step-number">System Resilience</div>
                    <span class="workflow-node-badge node-retrieve">DURABLE AGENT</span>
                    <p style="color: #cbd5e1; font-size: 0.9rem; margin-top: 8px;">
                        Fully inspectable decision graph with step-by-step state tracking and transparent auditability.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        # Show Workflow Runs
        runs = st.session_state.workflow_history
        c_sel, c_clr = st.columns([5, 1])
        with c_sel:
            run_options = [
                f"Run #{r['id']} ({r['timestamp']}): \"{r['query'][:55]}{'...' if len(r['query']) > 55 else ''}\""
                for r in reversed(runs)
            ]
            selected_label = st.selectbox("Select Execution Trace to Inspect:", run_options, index=0)
            selected_id = int(selected_label.split("Run #")[1].split(" ")[0])
            run = next(r for r in runs if r["id"] == selected_id)
        with c_clr:
            if st.button("Clear Traces", help="Clear workflow history"):
                st.session_state.workflow_history = []
                st.rerun()

        # Top Metric Row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Execution Mode", run["mode"])
        with m2:
            st.metric("Graph Steps", f"{len(run.get('steps', []))} Nodes")
        with m3:
            is_gr = run.get("is_grounded", True)
            grounded_label = "Grounded & Verified ✅" if is_gr else "Refinement Required ⚠️"
            st.metric("Fact Groundedness", grounded_label)
        with m4:
            has_rewritten = bool(run.get("rewritten_query"))
            st.metric("Query Rewritten", "Yes 🔄" if has_rewritten else "No (Direct) 🎯")

        # Query Transformation Highlight (if rewritten)
        if run.get("rewritten_query"):
            st.markdown(
                f"""
                <div style="background: rgba(245, 158, 11, 0.12); border: 1px solid rgba(245, 158, 11, 0.35); border-radius: 10px; padding: 14px 18px; margin: 14px 0;">
                    <div style="font-weight: 700; color: #fbbf24; margin-bottom: 6px;">🔄 Query Reformulation Triggered by Agent</div>
                    <div style="font-size: 0.9rem; color: #cbd5e1; margin-bottom: 4px;"><b>Original Question:</b> <code>{run['query']}</code></div>
                    <div style="font-size: 0.9rem; color: #38bdf8;"><b>Refined Vector Search Query:</b> <code>{run['rewritten_query']}</code></div>
                    <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 6px;"><i>The agent determined that initial retrieved chunks were incomplete or ambiguous, prompting semantic query reformulation.</i></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Timeline of executed steps
        st.markdown(f"#### 📋 Step-by-Step Execution Graph ({len(run.get('steps', []))} Steps)")
        
        badge_class_map = {
            "retrieve": "node-retrieve",
            "grade_documents": "node-grade",
            "rewrite_query": "node-rewrite",
            "generate": "node-generate",
            "check_groundedness": "node-groundedness",
            "agent_error": "node-error",
        }
        
        icon_map = {
            "retrieve": "🔍",
            "grade_documents": "⚖️",
            "rewrite_query": "🔄",
            "generate": "⚡",
            "check_groundedness": "🛡️",
            "agent_error": "⚠️",
        }

        for idx, s in enumerate(run.get("steps", [])):
            node_key = s.get("node", "step").lower()
            node_badge = s.get("node", "step").upper()
            badge_class = badge_class_map.get(node_key, "node-retrieve")
            icon = icon_map.get(node_key, "✅")
            desc = s.get("description", "")
            details = s.get("details", {})
            status = s.get("status", "completed")

            st.markdown(
                f"""
                <div class="workflow-timeline-card">
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
                        <div>
                            <span style="font-size: 1.1rem; margin-right: 6px;">{icon}</span>
                            <span class="workflow-node-badge {badge_class}">{node_badge}</span>
                            <span style="font-weight: 600; font-size: 0.92rem; color: #e2e8f0;">Step {idx+1}</span>
                        </div>
                        <span style="font-size: 0.78rem; text-transform: uppercase; color: #10b981; font-weight: 600;">{status}</span>
                    </div>
                    <div style="color: #cbd5e1; font-size: 0.9rem; margin-left: 2rem;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Contextual breakdown for each node
            if node_key == "retrieve" and details:
                c1, c2 = st.columns(2)
                with c1:
                    st.caption(f"📊 **Chunks Found:** {details.get('chunks_found', 0)}")
                with c2:
                    sources = details.get("sources", [])
                    st.caption(f"📁 **Source Documents:** {', '.join(sources) if sources else 'None'}")
            elif node_key == "grade_documents" and details:
                c1, c2 = st.columns(2)
                with c1:
                    st.caption(f"✅ **Passed Relevance:** {details.get('passed', 0)}")
                with c2:
                    st.caption(f"❌ **Filtered Out:** {details.get('total', 0) - details.get('passed', 0)} of {details.get('total', 0)}")
            elif node_key == "rewrite_query" and details:
                st.caption(f"↳ **New Search Target:** `{details.get('rewritten_query', '')}`")
            elif node_key == "check_groundedness" and details:
                grounded = details.get("grounded", True)
                status_txt = "All claims verified against source context" if grounded else "Factuality check failed"
                st.caption(f"🛡️ **Groundedness Verification:** `{status_txt}`")

            if details:
                with st.expander(f"🔍 Inspect Node Payload ({node_badge})", expanded=False):
                    st.json(details)

        # Final Answer & Source Attribution Card in Workflow Tab
        st.markdown("#### 🎯 Synthesized Answer & Context Attribution")
        with st.container():
            st.markdown(
                f"""
                <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 10px; padding: 1.25rem; margin-top: 8px;">
                    <div style="font-size: 0.85rem; color: #94a3b8; font-weight: 600; margin-bottom: 8px;">FINAL GENERATED OUTPUT</div>
                    <div style="color: #f1f5f9; line-height: 1.6;">{run.get('answer', '')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        citations = run.get("citations", [])
        if citations:
            with st.expander(f"📚 Full Source Attribution & Context Chunks ({len(citations)})", expanded=False):
                for idx, cit in enumerate(citations):
                    src = cit.get("source", "Document")
                    score = cit.get("score")
                    score_text = f" • Similarity: {score*100:.1f}%" if score else ""
                    st.markdown(f"**[{idx+1}] {src}**{score_text}")
                    st.text(cit.get("text", ""))
with tab_upload:
    st.subheader("Upload Documents to Ingest")
    st.caption("Supports **PDF** (`.pdf`), **Word** (`.docx`), **Markdown** (`.md`), and **Plain Text** (`.txt`).")

    uploaded_files = st.file_uploader(
        "Choose documents",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )

    make_public = st.checkbox(
        "🌐 Make Public / Demo Document (Visible to Guests and All Users)",
        value=(current_user_id == "guest"),
        help="If checked, this document is added to the shared public library so any guest or user can ask questions about it.",
    )

    if uploaded_files:
        if st.button("🚀 Process & Index Documents", type="primary"):
            progress_bar = st.progress(0.0)
            total = len(uploaded_files)
            target_user_id = "public" if make_public else current_user_id

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
                        count = direct_document_ingest(saved_path, user_id=target_user_id)
                        status.update(label=f"Successfully indexed '{up_file.name}' ({count} chunks)!", state="complete")

                    progress_bar.progress((i + 1) / total)

            st.success(f"Finished processing {total} document(s)!")
            time.sleep(1)
            st.rerun()
with tab_inspector:
    st.subheader("Inspect Indexed Vectors & Payloads")
    sources = get_vector_store().list_sources(user_id=current_user_id)

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
