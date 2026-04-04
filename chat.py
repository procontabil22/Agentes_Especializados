"""
app/chat.py — RAG: embedding da pergunta → busca Supabase → LLM → resposta
Clientes inicializados lazy.
"""
from functools import lru_cache
from typing import Optional

from loguru import logger

from app.agents import AgentConfig, get_agent
from config.settings import settings


@lru_cache(maxsize=1)
def _supabase():
    from supabase import create_client
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)


@lru_cache(maxsize=1)
def _embeddings():
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        dimensions=settings.EMBEDDING_DIMENSIONS,
        openai_api_key=settings.OPENAI_API_KEY,
    )


def _get_llm():
    """Usa Claude se ANTHROPIC_API_KEY configurada, senão GPT-4o-mini."""
    if settings.ANTHROPIC_API_KEY:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            anthropic_api_key=settings.ANTHROPIC_API_KEY,
            max_tokens=2048,
        )
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=settings.OPENAI_API_KEY,
        max_tokens=2048,
    )


def _retrieve(agent: AgentConfig, question: str) -> list[dict]:
    q_vec = _embeddings().embed_query(question)
    resp = _supabase().rpc("match_documents", {
        "query_embedding": q_vec,
        "match_table":     agent.table_name,
        "match_threshold": agent.similarity_threshold,
        "match_count":     agent.k,
    }).execute()
    chunks = resp.data or []
    logger.debug(f"RAG: {len(chunks)} chunks para '{question[:60]}...'")
    return chunks


def _context_block(chunks: list[dict]) -> str:
    if not chunks:
        return "Nenhum documento relevante encontrado na base de conhecimento."
    parts = []
    for i, c in enumerate(chunks, 1):
        meta = c.get("metadata", {})
        src = meta.get("file_name", "desconhecido")
        sec = " > ".join(filter(None, [meta.get("h1",""), meta.get("h2","")]))
        hdr = f"[{i}] {src}" + (f" | {sec}" if sec else "")
        parts.append(f"{hdr}\n{c['content']}")
    return "\n\n---\n\n".join(parts)


async def ask_agent(agent_id: str, question: str, history: list[dict] = None) -> dict:
    agent = get_agent(agent_id)
    if not agent:
        return {"answer": f"Agente '{agent_id}' não encontrado.", "sources": [], "agent": agent_id}

    chunks  = _retrieve(agent, question)
    context = _context_block(chunks)

    system = f"""{agent.system_prompt}

═══════════════════════════════════════
BASE DE CONHECIMENTO (use como referência principal):

{context}
═══════════════════════════════════════

Responda com base no contexto acima. Se não for suficiente, informe claramente."""

    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    messages = [SystemMessage(content=system)]
    for msg in (history or []):
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=question))

    response = await _get_llm().ainvoke(messages)

    sources, seen = [], set()
    for c in chunks:
        meta = c.get("metadata", {})
        fname = meta.get("file_name", "")
        if fname and fname not in seen:
            seen.add(fname)
            sources.append({
                "file": fname,
                "section": " > ".join(filter(None, [meta.get("h1",""), meta.get("h2","")])),
                "score": round(c.get("similarity", 0), 3),
            })

    return {"answer": response.content, "sources": sources,
            "agent": agent.name, "agent_id": agent_id}
