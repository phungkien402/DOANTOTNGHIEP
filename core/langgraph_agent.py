"""
LangGraph Agent Orchestrator — uses LLM Orchestrator for routing decisions.

Graph flow:
  QueryAnalyzer → ToolRouter → FastRetriever → Orchestrator
                                                    ↓
                                  action=answer  → FullRetriever → Synthesizer → Generator
                                  action=clarify → END (return clarify_message)
                                  action=ticket  → TicketCreator
              → ChatFallback

Public API: run(message, session_history) -> Answer
Same signature as pipeline.py for drop-in replacement.

Run standalone: python3 core/langgraph_agent.py
"""

import sys
import time
from pathlib import Path
from typing import TypedDict, Optional, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from langgraph.graph import StateGraph, END

from config import CONFIDENCE_THRESHOLD, MAINTENANCE_MODE, RETRIEVER_TOP_K, RERANKER_TOP_N
from core.models import Message, Answer, RetrievedChunk
from core.intent_guard import classify, chat_fallback
from core.tools.create_ticket import save_ticket
from core.tools.search_knowledge import search_knowledge
from core.knowledge_store import rebuild_index
from core import generator, confidence, retriever, reranker
from core.abbreviations import expand_abbreviations
from core.langfuse_tracer import new_trace, log_span, end_trace
from core.trace_logger import start_trace as tl_start, log_event as tl_event, finish_trace as tl_finish

# Rebuild knowledge index at startup so _index.json is always fresh
rebuild_index()


# --- Session manager injection (set from api/routes.py to avoid circular imports) ---

_session_mgr = None


def set_session_manager(mgr):
    """Inject the SessionManager instance from api/routes.py."""
    global _session_mgr
    _session_mgr = mgr
    print("[AGENT] SessionManager injected")


# --- State schema ---

class AgentState(TypedDict):
    query: str                    # original user query
    is_ehc_related: bool
    intent: str                   # "search_faq" | "create_ticket" | "chat_fallback" | "clarify"
    rewritten_query: str
    tool_called: str              # actual tool that ran
    chunks: list                  # list of RetrievedChunk (after full retrieve + rerank)
    fast_chunks: list             # top 3 chunks from fast retrieve
    confidence: float
    answer: str
    ticket_id: Optional[int]
    user_intent: Optional[str]    # intent description from orchestrator reasoning
    session_history: list         # conversation history
    session_id: str               # needed for session tracking
    tool: str                     # "search_faq" | "search_manual"
    knowledge_topic: str          # stem of knowledge file to load (or "")
    knowledge_content: str        # loaded knowledge file body (or "")
    lf_trace: Any                 # Langfuse trace object (or None)
    trace_id: str                 # unique per query run (for trace_logger)


# --- Maintenance mode ---

_maintenance_mode: bool = MAINTENANCE_MODE

MAINTENANCE_MESSAGE = (
    "⚙️ Hệ thống đang bảo trì, vui lòng thử lại sau ít phút. "
    "Xin lỗi vì sự bất tiện này 🙏"
)


def set_maintenance_mode(enabled: bool):
    """Toggle maintenance mode at runtime."""
    global _maintenance_mode
    _maintenance_mode = enabled
    print(f"[AGENT] Maintenance mode: {'ON' if enabled else 'OFF'}")


def is_maintenance_mode() -> bool:
    """Check if maintenance mode is active."""
    return _maintenance_mode


# --- Node functions ---

def node_query_analyzer(state: AgentState) -> dict:
    """Classify query: EHC-related or off-topic."""
    query = state["query"]
    session_id = state.get("session_id", "")
    trace_id = state.get("trace_id", "")
    print(f"\n[AGENT] Node: QueryAnalyzer | query=\"{query}\"")

    t_start = time.time()

    # Bypass classifier if mid-clarification — let Orchestrator handle via history
    if _session_mgr and _session_mgr.is_awaiting_clarification(session_id):
        print(f"[AGENT] Classifier: BYPASS (awaiting_clarification=True)")
        elapsed = (time.time() - t_start) * 1000
        tl_event(trace_id, "IntentGuard", "decision", {
            "result": True, "bypass": True, "duration_ms": round(elapsed, 1),
        }, duration_ms=round(elapsed, 1))
        return {
            "is_ehc_related": True,
            "intent": "search_faq",
        }

    is_off_topic = classify(query)
    elapsed = (time.time() - t_start) * 1000

    is_ehc = not is_off_topic
    tl_event(trace_id, "IntentGuard", "decision", {
        "result": is_ehc, "query": query, "duration_ms": round(elapsed, 1),
    }, duration_ms=round(elapsed, 1))

    if is_off_topic:
        print(f"[AGENT] Classifier: NO (off-topic)")
        return {
            "is_ehc_related": False,
            "intent": "chat_fallback",
        }
    else:
        print(f"[AGENT] Classifier: YES (EHC-related)")
        return {
            "is_ehc_related": True,
            "intent": "search_faq",
        }


def node_tool_router(state: AgentState) -> dict:
    """Route to the appropriate tool based on intent."""
    intent = state["intent"]
    print(f"[AGENT] Node: ToolRouter | intent={intent}")

    if intent == "create_ticket":
        return {"tool_called": "ticket_creator"}
    else:
        # Default: fast retrieve for all EHC-related queries
        return {"tool_called": "fast_retriever"}


def node_fast_retriever(state: AgentState) -> dict:
    """Fast retrieve top chunks from BOTH collections for orchestrator context."""
    query = state["query"]
    session_id = state.get("session_id", "")
    trace_id = state.get("trace_id", "")
    print(f"[AGENT] Node: FastRetriever | query=\"{query}\"")

    t_start = time.time()

    # Reuse saved fast_chunks from clarification turn if available
    if _session_mgr:
        saved = _session_mgr.get_fast_chunks(session_id)
        if saved:
            print(f"[AGENT] Node: FastRetriever | reusing {len(saved)} saved chunks")
            elapsed = (time.time() - t_start) * 1000
            tl_event(trace_id, "FastRetriever", "end", {
                "chunks": [{"subject": c.metadata.get("subject", "")[:60], "score": round(c.score, 4)} for c in saved],
                "reused": True,
                "duration_ms": round(elapsed, 1),
            }, duration_ms=round(elapsed, 1))
            return {"fast_chunks": saved}

    # Expand abbreviations before retrieval
    expanded = expand_abbreviations(query)

    # Query both collections
    from core.tools.search_manual import _retrieve_manual
    faq_chunks = retriever.retrieve(expanded, top_k=2)
    manual_chunks = _retrieve_manual(expanded, top_k=2)

    # Merge and sort by score, take top 4
    all_chunks = sorted(faq_chunks + manual_chunks, key=lambda c: c.score, reverse=True)[:4]

    for i, c in enumerate(all_chunks, 1):
        src = c.metadata.get("source", "faq")
        print(f"[RETRIEVER] #{i} score={c.score:.3f} [{src}] | {c.metadata.get('subject','')[:60]}")

    elapsed = (time.time() - t_start) * 1000
    tl_event(trace_id, "FastRetriever", "end", {
        "chunks": [{"subject": c.metadata.get("subject", "")[:60], "score": round(c.score, 4)} for c in all_chunks],
        "duration_ms": round(elapsed, 1),
    }, duration_ms=round(elapsed, 1))

    print(f"[AGENT] Node: FastRetriever | {len(all_chunks)} chunks (faq+manual)")
    return {"fast_chunks": all_chunks}


def node_orchestrator(state: AgentState) -> dict:
    """LLM Orchestrator — decides action based on query + fast_chunks + history."""
    query = state["query"]
    fast_chunks = state.get("fast_chunks", [])
    session_history = state.get("session_history", [])
    session_id = state.get("session_id", "")
    trace_id = state.get("trace_id", "")
    lf_trace = state.get("lf_trace")
    print(f"[AGENT] Node: Orchestrator | query=\"{query}\"")

    from core.orchestrator import orchestrate

    t_orch_start = time.time()
    result = orchestrate(
        query=query,
        fast_chunks=fast_chunks,
        session_history=session_history,
    )

    action = result["action"]
    search_query = result.get("search_query", query)
    clarify_msg = result.get("clarify_message", "")
    reasoning = result.get("reasoning", "")

    elapsed = (time.time() - t_orch_start) * 1000

    log_span(
        lf_trace,
        "Orchestrator",
        input_data={"query": query, "fast_chunks": len(fast_chunks)},
        output_data={"action": action, "tool": result.get("tool", ""), "reasoning": reasoning[:100]},
        start_time=t_orch_start,
    )

    tl_event(trace_id, "Orchestrator", "decision", {
        "action": action,
        "tool": result.get("tool", ""),
        "knowledge_topic": result.get("knowledge_topic", ""),
        "reasoning": reasoning,
        "search_query": search_query,
        "clarify_message": clarify_msg[:200] if clarify_msg else "",
        "duration_ms": round(elapsed, 1),
    }, duration_ms=round(elapsed, 1))

    print(f"[AGENT] Node: Orchestrator | action={action} | search_query=\"{search_query}\"")

    if action == "clarify":
        # Set flag so next turn bypasses classifier
        if _session_mgr:
            _session_mgr.set_awaiting_clarification(session_id, True)
            _session_mgr.set_fast_chunks(session_id, fast_chunks)
        return {
            "answer": clarify_msg,
            "intent": "clarify",
            "tool_called": "clarifier",
            "rewritten_query": search_query,
        }
    elif action == "ticket":
        # Clear clarification state
        if _session_mgr:
            _session_mgr.set_awaiting_clarification(session_id, False)
            _session_mgr.set_fast_chunks(session_id, [])
        return {
            "intent": "create_ticket",
            "tool_called": "ticket_creator",
            "rewritten_query": search_query,
        }
    else:  # answer
        # Clear clarification state
        if _session_mgr:
            _session_mgr.set_awaiting_clarification(session_id, False)
            _session_mgr.set_fast_chunks(session_id, [])
        return {
            "intent": "search_faq",
            "rewritten_query": search_query,
            "tool_called": "full_retriever",
            "tool": result.get("tool", "search_faq"),
            "knowledge_topic": result.get("knowledge_topic", ""),
        }


def node_full_retriever(state: AgentState) -> dict:
    """Full retrieve (top K) + rerank (top N) using the orchestrator's search_query.
    Also loads knowledge content if knowledge_topic was set by orchestrator.
    """
    rewritten = state.get("rewritten_query", state["query"])
    tool = state.get("tool", "search_faq")
    knowledge_topic = state.get("knowledge_topic", "")
    trace_id = state.get("trace_id", "")
    lf_trace = state.get("lf_trace")
    print(f"[AGENT] Node: FullRetriever | tool={tool} | knowledge_topic={knowledge_topic} | query=\"{rewritten}\"")

    t_ret_start = time.time()
    if tool == "search_manual":
        from core.tools.search_manual import search_manual
        ranked_chunks, top_score = search_manual(rewritten)
    else:
        chunks = retriever.retrieve(rewritten, top_k=RETRIEVER_TOP_K)
        if not chunks:
            print(f"[AGENT] Node: FullRetriever | no chunks retrieved")
            tl_event(trace_id, "FullRetriever", "end", {
                "top_score": 0.0, "chunks": [], "knowledge_loaded": False,
                "knowledge_topic": knowledge_topic, "duration_ms": 0.0,
            }, duration_ms=0.0)
            return {"chunks": [], "confidence": 0.0, "knowledge_content": ""}
        ranked_chunks = reranker.rerank(rewritten, chunks, top_n=RERANKER_TOP_N)
        top_score = ranked_chunks[0].score if ranked_chunks else 0.0

    # Load knowledge content if orchestrator requested it
    knowledge_content = ""
    if knowledge_topic:
        knowledge_content = search_knowledge(knowledge_topic)
        if knowledge_content.startswith("[Knowledge file"):
            # File not found — log and continue without it
            print(f"[AGENT] Node: FullRetriever | knowledge topic '{knowledge_topic}' not found, skipping")
            knowledge_content = ""

    elapsed = (time.time() - t_ret_start) * 1000

    log_span(
        lf_trace,
        "FullRetriever",
        input_data={"query": rewritten, "tool": tool, "knowledge_topic": knowledge_topic},
        output_data={"chunks": len(ranked_chunks), "top_score": round(top_score, 4), "has_knowledge": bool(knowledge_content)},
        start_time=t_ret_start,
    )

    tl_event(trace_id, "FullRetriever", "end", {
        "top_score": round(top_score, 4),
        "chunks": [{"subject": c.metadata.get("subject", "")[:60], "score": round(c.score, 4)} for c in ranked_chunks[:5]],
        "knowledge_loaded": bool(knowledge_content),
        "knowledge_topic": knowledge_topic,
        "duration_ms": round(elapsed, 1),
    }, duration_ms=round(elapsed, 1))

    print(f"[AGENT] Node: FullRetriever | tool={tool} | top_score={top_score:.4f} | knowledge={'yes' if knowledge_content else 'no'}")
    return {"chunks": ranked_chunks, "confidence": top_score, "knowledge_content": knowledge_content}


def node_synthesizer(state: AgentState) -> dict:
    """Check rerank confidence and decide: generate answer or create ticket."""
    chunks = state["chunks"]
    trace_id = state.get("trace_id", "")

    top_score = chunks[0].score if chunks else 0.0
    is_confident = chunks and confidence.is_confident(chunks[0], threshold=CONFIDENCE_THRESHOLD)

    route = "generator" if is_confident else "ticket_creator"
    tl_event(trace_id, "Synthesizer", "decision", {
        "confidence": round(top_score, 4),
        "threshold": CONFIDENCE_THRESHOLD,
        "route": route,
    })

    if is_confident:
        print(f"[AGENT] Node: Synthesizer | confidence={top_score:.4f} → CONFIDENT")
        return {"confidence": top_score, "intent": "search_faq"}
    else:
        print(f"[AGENT] Node: Synthesizer | confidence={top_score:.4f} → LOW → ticket")
        return {"confidence": top_score, "intent": "create_ticket"}


def node_generator(state: AgentState) -> dict:
    """Generate a grounded answer from retrieved chunks + optional knowledge content."""
    rewritten = state["rewritten_query"]
    chunks = state["chunks"]
    session_history = state.get("session_history", [])
    knowledge_content = state.get("knowledge_content", "")
    trace_id = state.get("trace_id", "")
    lf_trace = state.get("lf_trace")

    print(f"[AGENT] Node: Generator | chunks={len(chunks)} | knowledge={'yes' if knowledge_content else 'no'}")

    t_gen_start = time.time()
    try:
        answer_text = generator.generate(
            rewritten, chunks, session_history,
            knowledge_context=knowledge_content,
        )
    except Exception as e:
        print(f"[AGENT] Generator failed: {e}")
        answer_text = (
            "⚠️ Hệ thống AI đang bận hoặc đang khởi động lại, "
            "vui lòng thử lại sau 1–2 phút. Nếu vẫn lỗi, liên hệ bộ phận IT để kiểm tra server."
        )

    elapsed = (time.time() - t_gen_start) * 1000

    log_span(
        lf_trace,
        "Generator",
        input_data={"chunks": len(chunks), "query": rewritten},
        output_data={"answer_len": len(answer_text)},
        start_time=t_gen_start,
    )

    tl_event(trace_id, "Generator", "end", {
        "answer_chars": len(answer_text),
        "duration_ms": round(elapsed, 1),
    }, duration_ms=round(elapsed, 1))

    print(f"[AGENT] Node: Generator | answer_len={len(answer_text)}")
    return {"answer": answer_text}


def node_ticket_creator(state: AgentState) -> dict:
    """Create a ticket for unresolvable queries."""
    query = state["query"]
    user_intent = state.get("user_intent")
    trace_id = state.get("trace_id", "")
    print(f"[AGENT] Node: TicketCreator | query=\"{query}\"")

    ticket_id = save_ticket(
        query,
        user_intent=user_intent,
        rewritten_query=state.get("rewritten_query", ""),
        confidence=state.get("confidence", 0.0),
    )

    answer = (
        f"Mình đã ghi nhận vấn đề của bạn do vấn đề này chưa có trong cơ sở dữ liệu của mình (ticket #{ticket_id}). "
        "Vui lòng nhắn lại yêu cầu vào nhóm Zalo hỗ trợ để được nhân viên kỹ thuật giải đáp."
    )

    tl_event(trace_id, "Fallback", "info", {
        "reason": "confidence_below_threshold",
        "confidence": round(state.get("confidence", 0.0), 4),
        "ticket_id": ticket_id,
    })

    print(f"[AGENT] Node: TicketCreator | ticket_id={ticket_id}")
    return {
        "ticket_id": ticket_id,
        "answer": answer,
    }


def node_chat_fallback(state: AgentState) -> dict:
    """Generate a short polite off-topic response."""
    query = state["query"]
    trace_id = state.get("trace_id", "")
    print(f"[AGENT] Node: ChatFallback | query=\"{query}\"")

    t_start = time.time()
    answer = chat_fallback(query)
    elapsed = (time.time() - t_start) * 1000

    tl_event(trace_id, "ChatFallback", "end", {
        "answer": answer[:100],
        "duration_ms": round(elapsed, 1),
    }, duration_ms=round(elapsed, 1))

    print(f"[AGENT] Node: ChatFallback | answer=\"{answer}\"")
    return {"answer": answer}


# --- Graph wiring ---

graph = StateGraph(AgentState)

# Add nodes
graph.add_node("query_analyzer", node_query_analyzer)
graph.add_node("tool_router", node_tool_router)
graph.add_node("fast_retriever", node_fast_retriever)
graph.add_node("orchestrator", node_orchestrator)
graph.add_node("full_retriever", node_full_retriever)
graph.add_node("synthesizer", node_synthesizer)
graph.add_node("generator", node_generator)
graph.add_node("ticket_creator", node_ticket_creator)
graph.add_node("chat_fallback", node_chat_fallback)

# Set entry point
graph.set_entry_point("query_analyzer")

# Conditional edges
graph.add_conditional_edges(
    "query_analyzer",
    lambda s: "chat_fallback" if not s["is_ehc_related"] else "tool_router"
)
graph.add_conditional_edges(
    "tool_router",
    lambda s: s["tool_called"]  # "fast_retriever" or "ticket_creator"
)

# Fast retriever always goes to orchestrator
graph.add_edge("fast_retriever", "orchestrator")

# Orchestrator routes based on action
graph.add_conditional_edges(
    "orchestrator",
    lambda s: {
        "search_faq": "full_retriever",
        "clarify": END,
        "create_ticket": "ticket_creator",
    }.get(s["intent"], "full_retriever")
)

# Full retriever → synthesizer
graph.add_edge("full_retriever", "synthesizer")

# Synthesizer routes: confident → generator, low → ticket
graph.add_conditional_edges(
    "synthesizer",
    lambda s: "generator" if s["intent"] == "search_faq" else "ticket_creator"
)

# Terminal edges
graph.add_edge("generator", END)
graph.add_edge("ticket_creator", END)
graph.add_edge("chat_fallback", END)

# Compile the graph
app = graph.compile()


# --- Public API (same signature as pipeline.py) ---

def run(message: Message, session_history: list) -> Answer:
    """
    Drop-in replacement for pipeline.run().
    Accepts a Message object and session history, returns an Answer.
    """
    # Short-circuit if maintenance mode is active
    if _maintenance_mode:
        print(f"[AGENT] Maintenance mode active — returning maintenance message")
        return Answer(
            text=MAINTENANCE_MESSAGE,
            confidence=0.0,
            source_chunks=[],
            is_fallback=True,
            rewritten_question="",
        )

    print(f"\n{'='*60}")
    print(f"[AGENT] Input: \"{message.text}\"")
    print(f"{'='*60}")

    lf_trace = new_trace(query=message.text, session_id=message.session_id)

    # Generate unique trace_id for trace_logger
    trace_id = f"{message.user_id}-{int(time.time()*1000)}"
    tl_start(trace_id, message.text, user_id=message.user_id, platform=message.platform)

    initial_state: AgentState = {
        "query": message.text,
        "is_ehc_related": False,
        "intent": "search_faq",
        "rewritten_query": "",
        "tool_called": "",
        "chunks": [],
        "fast_chunks": [],
        "confidence": 0.0,
        "answer": "",
        "ticket_id": None,
        "user_intent": None,
        "session_history": session_history,
        "session_id": message.session_id,
        "tool": "search_faq",
        "knowledge_topic": "",
        "knowledge_content": "",
        "lf_trace": lf_trace,
        "trace_id": trace_id,
    }

    result = app.invoke(initial_state)

    # Build Answer object
    chunks = result.get("chunks", [])
    conf = result.get("confidence", 0.0)

    # Cast float32 → float for JSON serialization (RQ stores result in Redis)
    for chunk in chunks:
        chunk.score = float(chunk.score)

    is_fallback = result.get("intent") in ("chat_fallback", "create_ticket", "clarify")
    rewritten = result.get("rewritten_query", "")

    answer = Answer(
        text=result["answer"],
        confidence=float(conf),
        source_chunks=chunks,
        is_fallback=is_fallback,
        rewritten_question=rewritten,
    )

    tool_used = result.get("tool_called", result.get("intent", "unknown"))
    answered = conf >= CONFIDENCE_THRESHOLD and not is_fallback
    end_trace(lf_trace, tool=tool_used, confidence=conf, answered=answered)

    # Finish trace_logger trace
    tl_finish(trace_id, answer.text, is_fallback=answer.is_fallback)

    print(f"\n[AGENT] Done | tool={tool_used} confidence={conf:.4f}")
    return answer


if __name__ == "__main__":
    print("=== LangGraph Agent — Standalone Test ===\n")

    test_queries = [
        ("không in được", "Ambiguous → orchestrator should clarify"),
        ("không in được tài liệu khi chưa ký", "Clear → orchestrator should answer"),
        ("xin chào", "Off-topic → chat_fallback"),
    ]

    for query, description in test_queries:
        print(f"\n{'='*60}")
        print(f"TEST: {description}")
        print(f"{'='*60}")

        msg = Message(
            user_id="test", session_id="s1",
            text=query, timestamp=time.time(), platform="web"
        )
        answer = run(msg, [])
        print(f"\n  Bot: {answer.text}")
        print(f"  [confidence={answer.confidence:.2f} fallback={answer.is_fallback}]")

    print(f"\n{'='*60}")
    print("✓ All test queries completed.")
