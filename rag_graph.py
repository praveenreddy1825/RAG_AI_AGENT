import logging
from typing import TypedDict, List, Dict, Any, Optional

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from config import settings
from vector_db import QdrantStorage
from data_loader import embed_texts

logger = logging.getLogger("rag_agent.graph")

class GraphState(TypedDict):
    """Represents the shared mutable state across the Agentic RAG graph nodes."""
    question: str                        
    original_question: str                
    top_k: int                            
    documents: List[Dict[str, Any]]       
    graded_documents: List[Dict[str, Any]]
    generation: str                       
    steps: List[Dict[str, Any]]           
    loop_count: int                       
    is_grounded: bool                    

def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    api_key = settings.OPENAI_API_KEY
    if not api_key:
        logger.warning("OPENAI_API_KEY is not configured in settings or environment.")
    return ChatOpenAI(
        api_key=api_key,
        model=settings.OPENAI_LLM_MODEL,
        temperature=temperature,
    )

def retrieve_node(state: GraphState) -> Dict[str, Any]:
    query = state["question"]
    top_k = state.get("top_k", settings.DEFAULT_TOP_K)
    steps = list(state.get("steps", []))

    logger.info(f"[LangGraph: Retrieve] Searching Qdrant for: '{query}' (top_k={top_k})")
    
    try:
        query_vec = embed_texts([query])[0]
        store = QdrantStorage()
        results = store.search(query_vec, top_k=top_k)
        citations = results.get("citations", [])
    except Exception as e:
        logger.error(f"[LangGraph: Retrieve] Search failed: {e}")
        citations = []

    steps.append({
        "node": "retrieve",
        "status": "completed",
        "description": f"Retrieved {len(citations)} candidate chunk(s) from Qdrant vector storage.",
        "details": {
            "query": query,
            "chunks_found": len(citations),
            "sources": list({c.source for c in citations}) if citations else [],
        },
    })

    doc_dicts = [
        c.model_dump() if hasattr(c, "model_dump") else c
        for c in citations
    ]

    return {
        "documents": doc_dicts,
        "steps": steps,
    }

def grade_documents_node(state: GraphState) -> Dict[str, Any]:
    """
    Evaluates retrieved chunks for topical relevance to the query.
    Filters out irrelevant context to minimize noise and hallucination.
    """
    question = state["question"]
    documents = state.get("documents", [])
    steps = list(state.get("steps", []))

    if not documents:
        steps.append({
            "node": "grade_documents",
            "status": "completed",
            "description": "No candidate chunks were available to grade.",
            "details": {"passed": 0, "total": 0},
        })
        return {"graded_documents": [], "steps": steps}

    llm = get_llm(temperature=0.0)
    doc_summaries = "\n\n".join(
        f"[Chunk {idx+1}] (Source: {d.get('source', 'Unknown')}):\n{d.get('text', '')[:400]}..."
        for idx, d in enumerate(documents)
    )

    prompt = (
        f"You are a strict relevance evaluator for a Retrieval-Augmented Generation system.\n"
        f"User Question: \"{question}\"\n\n"
        f"Below are candidate document chunks retrieved from the vector database:\n"
        f"{doc_summaries}\n\n"
        f"Task: For each chunk from 1 to {len(documents)}, determine whether it contains information "
        f"relevant to answering the user question.\n"
        f"Output format: Output exactly one line per chunk, e.g.:\n"
        f"1: yes\n"
        f"2: no\n"
        f"Answer ONLY with the numbers and yes/no. Do not add explanations."
    )

    try:
        response = llm.invoke([
            SystemMessage(content="You are an expert document relevance evaluator. Return only the requested formatted lines."),
            HumanMessage(content=prompt),
        ])
        lines = response.content.strip().split("\n")
        relevance_map = {}
        for line in lines:
            if ":" in line:
                parts = line.split(":", 1)
                try:
                    chunk_num = int(parts[0].strip().replace("[", "").replace("]", "").replace("Chunk", "").strip())
                    is_relevant = "yes" in parts[1].strip().lower()
                    relevance_map[chunk_num] = is_relevant
                except ValueError:
                    continue

        graded_docs = []
        for idx, doc in enumerate(documents, start=1):
            if relevance_map.get(idx, True):  # Default to True if grading parsing failed for that index
                graded_docs.append(doc)

    except Exception as e:
        logger.warning(f"[LangGraph: Grade] Batch grading error: {e}. Keeping all retrieved documents.")
        graded_docs = documents

    steps.append({
        "node": "grade_documents",
        "status": "completed",
        "description": f"Relevance check: {len(graded_docs)} of {len(documents)} document chunk(s) passed grading.",
        "details": {
            "passed": len(graded_docs),
            "filtered_out": len(documents) - len(graded_docs),
        },
    })

    return {
        "graded_documents": graded_docs,
        "steps": steps,
    }

def rewrite_query_node(state: GraphState) -> Dict[str, Any]:
    """
    Transforms the user's query into an optimized version for semantic vector search
    when previous attempts produced zero relevant chunks.
    """
    current_query = state["question"]
    original_query = state["original_question"]
    loop_count = state.get("loop_count", 0)
    steps = list(state.get("steps", []))

    llm = get_llm(temperature=0.2)

    prompt = (
        f"You are an AI query optimizer for a semantic vector database.\n"
        f"The user asked: \"{original_query}\"\n"
        f"The previous search query was: \"{current_query}\"\n"
        f"However, this query failed to find relevant document chunks.\n\n"
        f"Task: Formulate an improved, clear, keywords-rich search query that captures the underlying intent "
        f"and vocabulary likely found in technical or corporate documentation.\n"
        f"Provide ONLY the rewritten query text. Do not add quotes or markdown."
    )

    try:
        response = llm.invoke([
            SystemMessage(content="You are a search query optimizer. Return only the rewritten search query string."),
            HumanMessage(content=prompt),
        ])
        rewritten = response.content.strip().strip('"').strip("'")
    except Exception as e:
        logger.warning(f"[LangGraph: Rewrite] Rewriting failed: {e}. Reusing original query.")
        rewritten = original_query

    new_loop_count = loop_count + 1

    steps.append({
        "node": "rewrite_query",
        "status": "completed",
        "description": f"Query Rewriter (Loop {new_loop_count}): Refined search query to '{rewritten}'.",
        "details": {
            "previous_query": current_query,
            "rewritten_query": rewritten,
            "loop": new_loop_count,
        },
    })

    return {
        "question": rewritten,
        "loop_count": new_loop_count,
        "steps": steps,
    }

def generate_node(state: GraphState) -> Dict[str, Any]:
    """Synthesizes a response strictly grounded in verified context chunks."""
    question = state["original_question"]
    docs = state.get("graded_documents") or state.get("documents") or []
    steps = list(state.get("steps", []))

    llm = get_llm(temperature=0.2)

    if not docs:
        fallback_msg = (
            f"I could not find relevant information in the uploaded documents to answer your question: "
            f"\"{question}\". Please verify that relevant files have been uploaded."
        )
        steps.append({
            "node": "generate",
            "status": "completed",
            "description": "No relevant documents available; generated transparent fallback notice.",
        })
        return {"generation": fallback_msg, "steps": steps}

    context_str = "\n\n".join(
        f"[Source {i+1}: {d.get('source', 'Unknown')}]:\n{d.get('text', '')}"
        for i, d in enumerate(docs)
    )

    prompt = (
        f"You are a precise, professional AI assistant. Answer the user's question using ONLY the provided context.\n"
        f"Cite your sources using [Source 1], [Source 2], etc., matching the provided context.\n"
        f"If the context does not contain sufficient details to answer, state what is known and mention what is missing.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    try:
        response = llm.invoke([
            SystemMessage(content="You are a helpful and truthful RAG assistant that answers questions using only provided context."),
            HumanMessage(content=prompt),
        ])
        answer = response.content.strip()
    except Exception as e:
        logger.error(f"[LangGraph: Generate] LLM inference failed: {e}")
        answer = f"Error generating answer: {str(e)}"

    steps.append({
        "node": "generate",
        "status": "completed",
        "description": f"Generated grounded response using {len(docs)} verified context source(s).",
    })

    return {
        "generation": answer,
        "steps": steps,
    }

def check_groundedness_node(state: GraphState) -> Dict[str, Any]:
    docs = state.get("graded_documents") or state.get("documents") or []
    answer = state.get("generation", "")
    steps = list(state.get("steps", []))

    if not docs or not answer or "could not find relevant information" in answer:
        steps.append({
            "node": "check_groundedness",
            "status": "skipped",
            "description": "Groundedness check skipped (empty or fallback answer).",
        })
        return {"is_grounded": True, "steps": steps}

    llm = get_llm(temperature=0.0)

    context_sample = "\n\n".join(d.get("text", "")[:500] for d in docs[:4])

    prompt = (
        f"You are a hallucination grader. Assess whether the following answer is fully supported by the provided facts.\n\n"
        f"Context Facts:\n{context_sample}\n\n"
        f"Proposed Answer:\n{answer}\n\n"
        f"Task: Is the answer factually grounded in the provided facts? Answer 'yes' or 'no'.\n"
        f"Answer ONLY 'yes' or 'no'."
    )

    try:
        response = llm.invoke([
            SystemMessage(content="You evaluate factual consistency. Answer only 'yes' or 'no'."),
            HumanMessage(content=prompt),
        ])
        is_grounded = "yes" in response.content.strip().lower()
    except Exception as e:
        logger.warning(f"[LangGraph: Hallucination Check] Evaluation failed: {e}. Defaulting to grounded.")
        is_grounded = True

    steps.append({
        "node": "check_groundedness",
        "status": "completed" if is_grounded else "flagged",
        "description": (
            "Verified that answer is factually grounded in source documents."
            if is_grounded
            else "Detected potential unsupported claims in generation."
        ),
        "details": {"is_grounded": is_grounded},
    })

    return {
        "is_grounded": is_grounded,
        "steps": steps,
    }

def route_after_grading(state: GraphState) -> str:
    graded_docs = state.get("graded_documents", [])
    loop_count = state.get("loop_count", 0)
    max_loops = settings.AGENTIC_RAG_MAX_LOOPS

    if graded_docs:
        logger.info("[LangGraph Route] Documents passed grading -> Proceeding to 'generate'")
        return "generate"

    if loop_count < max_loops:
        logger.info(f"[LangGraph Route] No relevant documents found ({loop_count}/{max_loops}) -> Routing to 'rewrite_query'")
        return "rewrite_query"

    logger.info("[LangGraph Route] Max loops reached -> Proceeding to 'generate' with best effort")
    return "generate"


def route_after_groundedness(state: GraphState) -> str:
    is_grounded = state.get("is_grounded", True)
    loop_count = state.get("loop_count", 0)
    max_loops = settings.AGENTIC_RAG_MAX_LOOPS

    if is_grounded or loop_count >= max_loops:
        return END

    logger.info(f"[LangGraph Route] Generation flagged as ungrounded -> Triggering query rewrite (loop {loop_count})")
    return "rewrite_query"

def build_rag_graph():
    workflow = StateGraph(GraphState)

    # Add Nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("check_groundedness", check_groundedness_node)

    # Set Entry Point
    workflow.set_entry_point("retrieve")

    # Connect Edges
    workflow.add_edge("retrieve", "grade_documents")
    
    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
        },
    )

    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("generate", "check_groundedness")

    workflow.add_conditional_edges(
        "check_groundedness",
        route_after_groundedness,
        {
            END: END,
            "rewrite_query": "rewrite_query",
        },
    )

    return workflow.compile()
_compiled_graph = None

def get_rag_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_rag_graph()
    return _compiled_graph

def run_agentic_rag(question: str, top_k: int = 5) -> Dict[str, Any]:
    graph = get_rag_graph()
    
    initial_state: GraphState = {
        "question": question.strip(),
        "original_question": question.strip(),
        "top_k": top_k,
        "documents": [],
        "graded_documents": [],
        "generation": "",
        "steps": [],
        "loop_count": 0,
        "is_grounded": True,
    }

    try:
        final_state = graph.invoke(initial_state)
    except Exception as e:
        err_str = str(e)
        logger.error(f"[LangGraph: Error] Execution failed: {err_str}")
        if "insufficient_quota" in err_str or "credit_balance_exhausted" in err_str or "429" in err_str:
            friendly_msg = (
                "⚠️ OpenAI API Quota Exhausted: Your current OpenAI API key has no remaining credits. "
                "Please add credits at https://platform.openai.com/settings/organization/billing or update OPENAI_API_KEY in your .env file."
            )
        elif "invalid_api_key" in err_str or "401" in err_str:
            friendly_msg = (
                "⚠️ Invalid OpenAI API Key: Please verify that OPENAI_API_KEY is correctly set in your .env file."
            )
        else:
            friendly_msg = f"⚠️ LangGraph execution error: {err_str}"

        return {
            "answer": friendly_msg,
            "sources": [],
            "num_contexts": 0,
            "citations": [],
            "steps": [{
                "node": "agent_error",
                "status": "failed",
                "description": friendly_msg,
                "details": {"error": err_str},
            }],
            "original_question": question,
            "rewritten_query": None,
            "is_grounded": False,
            "iterations": 1,
        }

    used_docs = final_state.get("graded_documents") or final_state.get("documents") or []
    sources = sorted(list({d.get("source") for d in used_docs if d.get("source")}))
    
    rewritten = final_state.get("question")
    has_rewritten = rewritten != question.strip()

    return {
        "answer": final_state.get("generation", ""),
        "sources": sources,
        "num_contexts": len(used_docs),
        "citations": used_docs,
        "steps": final_state.get("steps", []),
        "original_question": question,
        "rewritten_query": rewritten if has_rewritten else None,
        "is_grounded": final_state.get("is_grounded", True),
        "iterations": final_state.get("loop_count", 0) + 1,
    }
