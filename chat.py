"""
app/chat.py — RAG com Parent-Child Retrieval

Fluxo:
  1. embed(pergunta)
  2. match_documents → retorna N children mais similares
  3. coleta parent_ids únicos dos children
  4. busca conteúdo completo dos parents (contexto rico)
  5. monta prompt → LLM → resposta

Retrocompatibilidade:
  Chunks antigos (chunk_level='flat') são retornados diretamente pelo
  match_documents e usados como contexto (sem lookup de parent).
"""

from functools import lru_cache
from typing import Optional

from loguru import logger

from app.agents import AgentConfig, get_agent
from config.settings import settings


# ── Clientes lazy ─────────────────────────────────────────────────────────────

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


# ── Retrieval parent-child ────────────────────────────────────────────────────

def _search_children(agent: AgentConfig, query_vec: list[float]) -> list[dict]:
    """
    Chama match_documents no Supabase.
    Retorna children (chunk_level='child') + legados (chunk_level='flat').
    """
    resp = _supabase().rpc("match_documents", {
        "query_embedding": query_vec,
        "match_table":     agent.table_name,
        "match_threshold": agent.similarity_threshold,
        "match_count":     agent.k,
    }).execute()
    return resp.data or []


def _fetch_parents(table_name: str, parent_ids: list[str]) -> list[dict]:
    """
    Busca os parents completos pelos IDs coletados dos children.
    """
    if not parent_ids:
        return []
    resp = (
        _supabase()
        .table(table_name)
        .select("content, metadata, parent_id, file_name")
        .in_("parent_id", parent_ids)
        .eq("chunk_level", "parent")
        .execute()
    )
    return resp.data or []


def _retrieve(agent: AgentConfig, question: str) -> tuple[list[dict], list[dict]]:
    """
    Retorna (context_chunks, children_for_sources).

    context_chunks: conteúdo enviado ao LLM
        - Para novos docs: parents completos dos children encontrados
        - Para docs legados (flat): os próprios chunks retornados
    children_for_sources: usado apenas para montar a lista de sources na resposta
    """
    q_vec    = _embeddings().embed_query(question)
    children = _search_children(agent, q_vec)

    if not children:
        logger.debug(f"RAG: 0 resultados para '{question[:60]}'")
        return [], []

    # Separa children novos dos legados (flat)
    new_children  = [c for c in children if c.get("chunk_level") == "child"]
    flat_children = [c for c in children if c.get("chunk_level") != "child"]

    context_chunks: list[dict] = []

    # ── Novos: busca parents ──────────────────────────────────────────────────
    if new_children:
        # parent_ids únicos, preservando ordem de relevância
        seen_pids: set[str] = set()
        ordered_pids: list[str] = []
        for c in new_children:
            pid = c.get("parent_id", "")
            if pid and pid not in seen_pids:
                seen_pids.add(pid)
                ordered_pids.append(pid)

        parents = _fetch_parents(agent.table_name, ordered_pids)

        # Ordena parents pela ordem de relevância dos children
        pid_order = {pid: i for i, pid in enumerate(ordered_pids)}
        parents.sort(key=lambda p: pid_order.get(p.get("parent_id", ""), 999))

        context_chunks.extend(parents)
        logger.debug(
            f"RAG parent-child: {len(new_children)} children → "
            f"{len(parents)} parents para '{question[:60]}'"
        )

    # ── Legados: usa o chunk direto como contexto ─────────────────────────────
    if flat_children:
        context_chunks.extend(flat_children)
        logger.debug(
            f"RAG flat (legado): {len(flat_children)} chunks para '{question[:60]}'"
        )

    return context_chunks, children


# ── Montagem do bloco de contexto ─────────────────────────────────────────────

def _context_block(chunks: list[dict]) -> str:
    if not chunks:
        return "Nenhum documento relevante encontrado na base de conhecimento."

    parts = []
    for i, c in enumerate(chunks, 1):
        meta = c.get("metadata") or {}
        src  = meta.get("file_name") or c.get("file_name", "desconhecido")
        h1   = meta.get("h1", "")
        h2   = meta.get("h2", "")
        sec  = " > ".join(filter(None, [h1, h2]))
        hdr  = f"[{i}] {src}" + (f" | {sec}" if sec else "")

        if c.get("chunk_level") == "parent":
            hdr += " [contexto completo]"

        parts.append(f"{hdr}\n{c['content']}")

    return "\n\n---\n\n".join(parts)


# ── Montagem de sources para a resposta ───────────────────────────────────────

def _build_sources(children: list[dict]) -> list[dict]:
    """
    Usa os children (ou flat chunks) para gerar a lista de fontes,
    pois eles têm o score de similaridade.
    """
    sources: list[dict] = []
    seen: set[str] = set()

    for c in children:
        meta  = c.get("metadata") or {}
        fname = meta.get("file_name") or c.get("file_name", "")
        if not fname or fname in seen:
            continue
        seen.add(fname)
        sources.append({
            "file":    fname,
            "section": " > ".join(filter(None, [meta.get("h1", ""), meta.get("h2", "")])),
            "score":   round(c.get("similarity", 0), 3),
        })

    return sources


# ── Ponto de entrada público ──────────────────────────────────────────────────

async def ask_agent(
    agent_id: str,
    question: str,
    history: list[dict] = None,
) -> dict:
    agent = get_agent(agent_id)
    if not agent:
        return {
            "answer":   f"Agente '{agent_id}' não encontrado.",
            "sources":  [],
            "agent":    agent_id,
        }

    context_chunks, children = _retrieve(agent, question)
    context = _context_block(context_chunks)

    system = f"""{agent.system_prompt}

═══════════════════════════════════════
BASE DE CONHECIMENTO (use como referência principal):

{context}
═══════════════════════════════════════

Responda com base no contexto acima.
Se o contexto não for suficiente para responder com segurança, informe claramente."""

    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

    messages = [SystemMessage(content=system)]
    for msg in (history or []):
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=question))

    response = await _get_llm().ainvoke(messages)

    return {
        "answer":   response.content,
        "sources":  _build_sources(children),
        "agent":    agent.name,
        "agent_id": agent_id,
    }
