# ⚡ Enterprise Event-Driven RAG Agent Application

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/Agentic%20AI-LangGraph%20CRAG-1C3C3C.svg)](https://langchain-ai.github.io/langgraph/)
[![Groq](https://img.shields.io/badge/LLM%20Provider-Groq%20Multi--Model-f55036.svg)](https://groq.com/)
[![FastEmbed](https://img.shields.io/badge/Embeddings-FastEmbed%20(Local%20Free)-green.svg)](https://qdrant.github.io/fastembed/)
[![Qdrant](https://img.shields.io/badge/Vector%20DB-Qdrant%20(Local%20%26%20Cloud)-red.svg)](https://qdrant.tech/)
[![Inngest](https://img.shields.io/badge/Orchestration-Inngest%20Durable-blueviolet.svg)](https://www.inngest.com/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit%20Studio-FF4B4B.svg)](https://streamlit.io/)

A production-grade, event-driven **Retrieval-Augmented Generation (RAG)** application engineered for high reliability, zero-cost local vectorization, multi-format document reasoning, and transparent agentic decision tracing.

Powered by a **Multi-Model Groq architecture** (`openai/gpt-oss-120b` generator + `openai/gpt-oss-20b` evaluator), **FastEmbed** local dense embeddings (`BAAI/bge-small-en-v1.5`), a self-reflective **LangGraph Corrective RAG (CRAG)** graph, **Inngest** durable execution workflows, **Qdrant** vector storage, and a modern multi-tab **Streamlit Studio**.

---

## 🏗️ Architecture Overview

The system decouples ingestion and retrieval into durable, observable workflows using Inngest event orchestration alongside an embedded or cloud-native Qdrant vector store.

```mermaid
graph TD
    User([User / Client]) -->|Upload PDF, DOCX, TXT, MD| UI[Streamlit UI / REST API]
    User -->|Chat / Question| UI
    
    subgraph Ingestion Pipeline [FastAPI & Inngest Ingestion]
        API[FastAPI Gateway] -->|Trigger: rag/ingest_pdf| EventBus[Inngest Event Bus]
        EventBus --> Step1[Step 1: Parse Document - LlamaIndex / pypdf]
        Step1 --> Step2[Step 2: Sentence-Aware Chunking]
        Step2 --> Step3[Step 3: FastEmbed Local 384-dim Embeddings]
        Step3 --> Step4[Step 4: Batch Upsert to Qdrant]
    end

    subgraph LangGraph Agentic CRAG [Self-Reflective Query Graph]
        UI -->|Query| Router{Conversational Intent?}
        Router -->|Greeting / Chit-Chat| FastGreeting([Instant Polite Response + Docs List])
        Router -->|Document Question| Node1[Node 1: Retrieve from Qdrant]
        Node1 --> Node2[Node 2: Grade Document Relevance - Evaluator LLM]
        Node2 -->|Irrelevant Chunks| Node3[Node 3: Conditional Query Rewriter]
        Node3 -->|Re-retrieve (Max 1 Loop)| Node1
        Node2 -->|Relevant Chunks| Node4[Node 4: Context-Grounded Generator - Generator LLM]
        Node4 --> Node5[Node 5: Groundedness & Fact Check - Evaluator LLM]
        Node5 -->|Grounded| FinalAnswer([Grounded Response + Sources + Audit Trace])
    end
    
    subgraph Storage Layer [Qdrant Vector Database]
        Step4 --> Qdrant[(Qdrant Vector DB\nEmbedded Disk or Cloud)]
        Node1 <--> Qdrant
    end
```

---

## 🧠 Multi-Model & Free-Tier Tech Stack

| Component | Technology | Rationale |
| :--- | :--- | :--- |
| **Generator LLM** | **Groq: `openai/gpt-oss-120b`** | High-reasoning model for synthesizing context-grounded final responses with source citations. |
| **Evaluator LLM** | **Groq: `openai/gpt-oss-20b`** | Ultra-fast model optimized for document grading, semantic query rewriting, and groundedness verification. |
| **Embeddings** | **FastEmbed: `BAAI/bge-small-en-v1.5`** | **100% Free, runs locally on CPU**. Produces dense 384-dimensional vectors with zero API rate limits and zero quota costs (eliminates OpenAI 429 errors). |
| **Vector Storage** | **Qdrant (Local Embedded Disk / Remote)** | Embedded disk storage in `./qdrant_storage` with thread-safe client caching, auto-healing collection dimension matching, and cloud support. |
| **Agentic Framework**| **LangGraph (CRAG)** | Cyclic state graph orchestrating retrieval, document relevance evaluation, query reformulation, and fact checks. |
| **Durable Execution**| **Inngest** | Background event orchestration with step memoization, automatic retries, and failure recovery. |
| **User Interface** | **Streamlit Studio** | Clean 4-tab dashboard separating conversational chat from detailed agent workflow traces. |
| **REST API** | **FastAPI + Uvicorn** | High-performance asynchronous API with OpenAPI Swagger documentation. |

---

## 🖥️ Four-Tab Streamlit Dashboard

The web dashboard is organized into four dedicated tabs to keep the user experience clean while providing deep engineering visibility:

```
┌────────────────────────────────────────────────────────────────────────────────┐
│  💬 Response & Chat  │  🧠 Agent Workflow  │  📤 Document Ingestion  │  🔍 Inspector  │
└────────────────────────────────────────────────────────────────────────────────┘
```

1. **`💬 Response & Chat`**:
   - **Clean conversational view**: Displays only the user prompt and final synthesized answer.
   - **Compact citations**: Collapsible source badges displaying document names, chunk indexes, and similarity match percentages.
   - **Clear History button**: Quick button to reset conversation state.
   - **Uncluttered interface**: Debug logs, raw JSON payloads, and reasoning traces are routed to the dedicated Workflow tab.

2. **`🧠 Agent Workflow` (Reasoning Studio)**:
   - **Execution Trace Selector**: Switch between historical query traces to inspect any past agent decision.
   - **Executive KPI Metrics**: Execution Mode, Graph Steps Executed, Groundedness Status (`Grounded & Verified ✅`), and Query Formulation (`Rewritten 🔄` vs `Direct Match 🎯`).
   - **Query Reformulation Diff Box**: Side-by-side comparison of `Original Question` vs `Refined Vector Search Query` whenever the agent re-targets retrieval.
   - **Step-by-Step Interactive Timeline**: Color-coded cards for each node (`RETRIEVE`, `GRADE_DOCUMENTS`, `REWRITE_QUERY`, `GENERATE`, `CHECK_GROUNDEDNESS`) with pass/fail ratios and expandable raw JSON payloads.
   - **Architectural Blueprint**: When no query has been run yet, displays an interactive visual guide explaining the 5-node CRAG lifecycle.

3. **`📤 Document Ingestion`**:
   - Drag-and-drop batch upload for **PDF (`.pdf`)**, **Word (`.docx`)**, **Markdown (`.md`)**, and **Plain Text (`.txt`)**.
   - Real-time progress bars showing chunk splitting, embedding generation, and vector indexing.

4. **`🔍 Document Inspector`**:
   - Live inspection of Qdrant vector storage: explore chunks, stored text, point IDs, and custom metadata for any indexed document.

---

## 🔬 Agent Evaluation & Self-Reflection Framework

The application implements a multi-stage **Corrective RAG (CRAG)** evaluation pipeline to ensure high accuracy and prevent hallucinations:

```
  [User Query]
       │
       ▼
  ┌───────────────┐
  │ Fast-Path     │──(Is Greeting?)──► [Instant Response: Greeting + Document Metadata]
  │ Intent Filter │
  └───────────────┘
       │ (Document Question)
       ▼
  ┌───────────────┐
  │ Vector Search │◄────────────────────────────────────────┐
  └───────────────┘                                         │
       │                                                    │
       ▼                                                    │
  ┌──────────────────┐                                      │
  │ Relevance Grader │──(All Irrelevant & Loop < 1)──► [Query Rewriter]
  └──────────────────┘
       │ (At least 1 chunk passed relevance)
       ▼
  ┌──────────────────┐
  │ Grounded Answer  │
  │ Synthesis        │
  └──────────────────┘
       │
       ▼
  ┌──────────────────┐
  │ Groundedness &   │──(Factually Consistent)──► [Final Response + Sources + Audit Trace]
  │ Fact Check       │
  └──────────────────┘
```

### 1. Conversational Intent Fast-Path
- **Purpose**: Classifies user queries before vector search.
- **Mechanism**: Conversational greetings (`"hi"`, `"hello"`, `"who are you"`, `"help"`) bypass vector database retrieval entirely.
- **Benefit**: Responds **instantly (< 0.1s)** with a polite introduction and lists the currently indexed documents, preventing the agent from getting trapped in unnecessary document search loops.

### 2. Document Relevance Grading (`grade_documents_node`)
- **Purpose**: Filters out retrieved chunks that are noisy, misleading, or irrelevant to the question.
- **Mechanism**: The evaluator LLM (`openai/gpt-oss-20b`) evaluates each candidate chunk individually and assigns a binary relevance grade (`yes` / `no`).
- **Benefit**: Ensures only relevant context enters the generator prompt, reducing context window clutter and hallucination risks.

### 3. Controlled Query Reformulation (`rewrite_query_node`)
- **Purpose**: Rescues failed retrieval queries where vocabulary mismatch prevents finding relevant chunks.
- **Mechanism**: If all candidate chunks fail relevance grading, the evaluator analyzes the original query and previous search query to formulate an improved, keywords-rich vector search query.
- **Loop Boundary**: Capped at **`AGENTIC_RAG_MAX_LOOPS = 1`** with early-exit guards to prevent latency degradation.

### 4. Hallucination & Groundedness Verification (`check_groundedness_node`)
- **Purpose**: Validates factual consistency between the generated response and source document chunks.
- **Mechanism**: Evaluates whether every statement made in the answer is factually grounded in the source text.
- **Benefit**: Flags unsupported assertions and guarantees traceable, truthful answers.

---

## 🧪 Verified Types of Questions

The RAG agent has been verified and benchmarked against four primary categories of user interactions:

| Question Category | Typical User Prompts | Agent Workflow & Evaluation Behavior |
| :--- | :--- | :--- |
| **1. Conversational Chit-Chat & Greetings** | • `"hi"`<br>• `"hello there"`<br>• `"who are you?"`<br>• `"help"` | • **Evaluation**: Intent filter identifies conversational greeting.<br>• **Execution**: Bypasses vector retrieval loop immediately.<br>• **Output**: Responds in **< 0.1s** with a warm introduction, lists indexed documents (`23je0265_1 (6).pdf`), and invites questions. |
| **2. Factual & Specific Document Inquiries** | • `"What is the roll number mentioned in the document?"`<br>• `"What are the key policy deadlines?"`<br>• `"What eligibility criteria are listed?"` | • **Evaluation**: Retrieves top-k chunks, evaluator grades chunks as relevant.<br>• **Execution**: Generator synthesizes an exact answer citing `[Source 1]`.<br>• **Output**: Accurate factual answer with match percentage and verified groundedness badge. |
| **3. Document Summaries & Overview Requests** | • `"Summarize this document"`<br>• `"What is this PDF about?"`<br>• `"Give me the main takeaways."` | • **Evaluation**: Retrieves representative document chunks across sections.<br>• **Execution**: Chunks pass topical relevance; generator constructs a structured executive overview.<br>• **Output**: Comprehensive summary with source attribution. |
| **4. Out-of-Domain / Unanswerable Questions** | • `"What is the weather in Tokyo?"`<br>• `"Who won the 2024 World Cup?"`<br>• (Questions unrelated to indexed docs) | • **Evaluation**: Evaluator grades candidate chunks as irrelevant (`no`).<br>• **Execution**: Reformulates query once, confirms zero relevant chunks, and exits cleanly.<br>• **Output**: Transparent, honest fallback: *"I could not find relevant information in the uploaded documents to answer your question... Please verify relevant files are uploaded."* (Zero hallucinations). |

---

## 📁 Project Structure

```
RAG_AGENT_APP/
├── config.py                 # Centralized configuration & environment loader
├── custom_types.py           # Pydantic models (chunks, citations, queries, agent steps)
├── data_loader.py            # Multi-format doc parser (PDF, DOCX, TXT, MD) & FastEmbed embedder
├── vector_db.py              # Production Qdrant storage manager (singleton client cache)
├── rag_graph.py              # LangGraph Agentic CRAG engine (grading, rewrite, groundedness)
├── main.py                   # FastAPI application + Inngest durable workflows
├── streamlit_app.py          # Modern Streamlit studio (separated Chat & Workflow tabs)
├── pyproject.toml            # Project packaging & dependency specifications
├── requirements.txt          # Pinned dependency requirements
├── .env.example              # Environment variables template
├── uploads/                  # Uploaded document staging directory
├── qdrant_storage/           # Local Qdrant embedded collection storage
└── README.md                 # Complete system documentation
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- **Python 3.11, 3.12, or 3.13**
- A free **[Groq API Key](https://console.groq.com/)** (for ultra-fast free-tier LLM generation & evaluation)
- *(Optional)* [OpenAI API Key](https://platform.openai.com/) if using OpenAI models
- *(Optional)* Node.js & npm if using Inngest local dev server

### 2. Environment Setup

Clone or open the repository, then create a virtual environment using `uv` (recommended) or standard `venv`:

```bash
# Using uv (fastest)
uv venv .venv

# Activate on Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate on macOS / Linux
source .venv/bin/activate
```

Install dependencies:

```bash
# Using uv
uv pip install -r requirements.txt

# Or using standard pip
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy the `.env.example` file to `.env`:

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```ini
# Embedding Settings (100% Free, runs locally, no API key needed)
EMBEDDING_PROVIDER=fastembed
FASTEMBED_MODEL=BAAI/bge-small-en-v1.5
EMBEDDING_DIM=384

# Multi-Model LLM Configuration (Groq Free Tier)
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_GENERATION_MODEL=openai/gpt-oss-120b
GROQ_EVALUATION_MODEL=openai/gpt-oss-20b

# Qdrant Vector DB Settings (Embedded Local Disk)
QDRANT_MODE=local
QDRANT_STORAGE_PATH=./qdrant_storage
QDRANT_COLLECTION_NAME=rag_documents

# LangGraph CRAG Loop Limit
AGENTIC_RAG_MAX_LOOPS=1
```

---

## 🏃 Running the Application

### Option A: Launch the Streamlit Studio (Recommended)

Run the full interactive web application:

```bash
uv run streamlit run streamlit_app.py
```
*(Or: `streamlit run streamlit_app.py`)*

- Open **[http://localhost:8501](http://localhost:8501)** in your browser.
- Go to the **`📤 Document Ingestion`** tab to upload and index documents.
- Use the **`💬 Response & Chat`** tab to ask questions and receive clean answers with sources.
- Switch to the **`🧠 Agent Workflow`** tab to inspect the step-by-step CRAG decision graph!

---

### Option B: Launch Full Inngest Orchestration & FastAPI Backend

For distributed event-driven execution and REST API access:

1. **Start the FastAPI Backend**:
   ```bash
   uv run uvicorn main:app --reload --port 8000
   ```
   Interactive Swagger API documentation: **[http://localhost:8000/docs](http://localhost:8000/docs)**.

2. **Start the Inngest Dev Server**:
   ```bash
   npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest
   ```
   Inngest Dev Dashboard: **[http://localhost:8288](http://localhost:8288)** to monitor durable events and step execution.

3. **Start the Streamlit Frontend**:
   ```bash
   uv run streamlit run streamlit_app.py
   ```
   Enable the **"Use Inngest Dev Server"** toggle in the sidebar to route queries through Inngest.

---

## 🔌 API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Health check endpoint returning Qdrant connectivity and mode |
| `GET` | `/api/v1/stats` | Returns total indexed vectors and list of document sources |
| `POST` | `/api/v1/upload` | Multipart document upload and vector indexing |
| `POST` | `/api/v1/query` | Standard RAG semantic search and answer generation |
| `POST` | `/api/v1/query/agentic` | Full LangGraph Agentic CRAG execution with step trace |
| `DELETE` | `/api/v1/documents/{source_id}` | Deletes all indexed chunks for a given document source |

---

## ⚙️ Configuration Reference

| Environment Variable | Default | Description |
| :--- | :--- | :--- |
| `LLM_PROVIDER` | `groq` | LLM provider: `groq` (ultra-fast free tier) or `openai` |
| `GROQ_API_KEY` | *(Required if Groq)* | Groq API secret key |
| `GROQ_GENERATION_MODEL` | `openai/gpt-oss-120b` | High-reasoning model for grounded answer synthesis |
| `GROQ_EVALUATION_MODEL` | `openai/gpt-oss-20b` | Fast model for document grading, query rewriting, and fact checks |
| `EMBEDDING_PROVIDER` | `fastembed` | Embedding provider: `fastembed` (local, free) or `openai` |
| `FASTEMBED_MODEL` | `BAAI/bge-small-en-v1.5` | FastEmbed embedding model name |
| `EMBEDDING_DIM` | `384` | Vector dimensionality (384 for BGE-small, 3072 for OpenAI large) |
| `QDRANT_MODE` | `local` | Storage mode: `local` (embedded disk) or `remote` (server / cloud) |
| `QDRANT_STORAGE_PATH` | `./qdrant_storage` | Local directory for embedded Qdrant database |
| `QDRANT_COLLECTION_NAME` | `rag_documents` | Collection name in Qdrant |
| `AGENTIC_RAG_MAX_LOOPS` | `1` | Maximum query reformulation loops permitted per question |
| `DEFAULT_TOP_K` | `5` | Default number of candidate chunks retrieved per query |
| `CHUNK_SIZE` | `1000` | Target character size per text chunk |
| `CHUNK_OVERLAP` | `200` | Overlap character count between adjacent chunks |

---

## 🛡️ License

Distributed under the MIT License. See `LICENSE` for more information.
