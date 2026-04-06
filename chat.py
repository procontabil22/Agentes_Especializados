"""
chat.py — RAG com Parent-Child Retrieval

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
import re

from loguru import logger

from agents import AgentConfig, get_agent  # ← flat import
from settings import settings              # ← flat import

# Padrão para detectar NCM na pergunta do usuário
_RE_NCM_QUERY = re.compile(
    r"\b(\d{4}[.\-]?\d{2}[.\-]?\d{2}[.\-]?\d{2})"   # NCM 10 dígitos com pontuação
    r"|\b(\d{4}[.\-]?\d{2}[.\-]?\d{2})"              # NCM 8 dígitos
    r"|\b(\d{4}[.\-]?\d{2})"                          # Posição 6 dígitos
    r"|\b(\d{4})\b"                                   # Capítulo 4 dígitos
)


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


@lru_cache(maxsize=1)
def _get_llm():
    """
    Seleciona o LLM disponível na seguinte ordem de prioridade:
      1. Anthropic (Claude)
      2. OpenAI (GPT)
      3. Google Gemini
      4. Grok (xAI)
      5. DeepSeek

    Usa o primeiro que tiver a variável de ambiente configurada.
    Cacheado com @lru_cache para evitar instanciar a cada requisição.
    """

    # 1. Anthropic
    if settings.ANTHROPIC_API_KEY:
        from langchain_anthropic import ChatAnthropic
        logger.info("LLM selecionado: Anthropic (Claude Haiku)")
        return ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            anthropic_api_key=settings.ANTHROPIC_API_KEY,
            max_tokens=2048,
        )

    # 2. OpenAI
    if settings.OPENAI_API_KEY:
        from langchain_openai import ChatOpenAI
        logger.info("LLM selecionado: OpenAI (GPT-4o-mini)")
        return ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=settings.OPENAI_API_KEY,
            max_tokens=2048,
        )

    # 3. Google Gemini
    if settings.GEMINI_API_KEY:
        from langchain_google_genai import ChatGoogleGenerativeAI
        logger.info("LLM selecionado: Google Gemini")
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=settings.GEMINI_API_KEY,
            max_output_tokens=2048,
        )

    # 4. Grok (xAI) — usa interface compatível com OpenAI
    if settings.GROK_API_KEY:
        from langchain_openai import ChatOpenAI
        logger.info("LLM selecionado: Grok (xAI)")
        return ChatOpenAI(
            model="grok-beta",
            openai_api_key=settings.GROK_API_KEY,
            openai_api_base="https://api.x.ai/v1",
            max_tokens=2048,
        )

    # 5. DeepSeek — usa interface compatível com OpenAI
    if settings.DEEPSEEK_API_KEY:
        from langchain_openai import ChatOpenAI
        logger.info("LLM selecionado: DeepSeek")
        return ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=settings.DEEPSEEK_API_KEY,
            openai_api_base="https://api.deepseek.com/v1",
            max_tokens=2048,
        )

    raise RuntimeError(
        "Nenhuma chave de API de LLM configurada. "
        "Configure ao menos uma das variáveis: ANTHROPIC_API_KEY, OPENAI_API_KEY, "
        "GEMINI_API_KEY, GROK_API_KEY ou DEEPSEEK_API_KEY."
    )


def _get_llm_name() -> str:
    """Retorna o nome amigável do LLM ativo (para incluir na resposta da API)."""
    if settings.ANTHROPIC_API_KEY:
        return "Anthropic (Claude Haiku)"
    if settings.OPENAI_API_KEY:
        return "OpenAI (GPT-4o-mini)"
    if settings.GEMINI_API_KEY:
        return "Google Gemini"
    if settings.GROK_API_KEY:
        return "Grok (xAI)"
    if settings.DEEPSEEK_API_KEY:
        return "DeepSeek"
    return "desconhecido"


# ── Busca NCM exata ───────────────────────────────────────────────────────────

def _extract_ncms_from_question(question: str) -> list[str]:
    """Extrai códigos NCM da pergunta do usuário."""
    found = []
    for m in _RE_NCM_QUERY.finditer(question):
        ncm = next(g for g in m.groups() if g)
        # Remove pontuação para normalizar
        ncm_digits = re.sub(r"[.\-\s]", "", ncm)
        if len(ncm_digits) >= 4:
            found.append(ncm_digits)
    return list(dict.fromkeys(found))  # deduplica mantendo ordem


def _search_ncm(ncm_codes: list[str], limit: int = 10) -> list[dict]:
    """
    Busca registros NCM na tabela kb_ncm_fiscal.
    Retorna contexto estruturado com tratamento tributário por NCM.
    """
    if not ncm_codes:
        return []

    results = []
    for ncm in ncm_codes:
        try:
            resp = _supabase().rpc("search_ncm", {
                "p_ncm":   ncm,
                "p_uf":    "MA",
                "p_limit": limit,
            }).execute()
            if resp.data:
                results.extend(resp.data)
                logger.debug(f"  NCM {ncm}: {len(resp.data)} registros encontrados")
        except Exception as e:
            logger.debug(f"  ⚠ Busca NCM {ncm}: {e}")

    return results


def _ncm_context_block(ncm_results: list[dict]) -> str:
    """Formata resultados NCM como contexto estruturado para o LLM."""
    if not ncm_results:
        return ""

    parts = ["📦 TRATAMENTO TRIBUTÁRIO POR NCM (RICMS-MA e CONFAZ):\n"]
    seen = set()

    for r in ncm_results:
        key = f"{r.get('ncm')}_{r.get('beneficio')}"
        if key in seen:
            continue
        seen.add(key)

        linha = f"• NCM {r.get('ncm', '')}"
        if r.get("descricao"):
            linha += f" — {r['descricao']}"
        linha += f"\n  Tratamento: {r.get('beneficio', '').upper()}"
        if r.get("percentual"):
            linha += f" | {r['percentual']}"
        if r.get("base_calculo"):
            linha += f" | Base: {r['base_calculo']}"
        if r.get("condicao"):
            linha += f"\n  Condição: {r['condicao']}"
        if r.get("dispositivo"):
            linha += f"\n  Dispositivo: {r['dispositivo']}"
        if r.get("file_name"):
            linha += f"\n  Fonte: {r['file_name']}"
        parts.append(linha)

    return "\n\n".join(parts)


# ── Retrieval parent-child ────────────────────────────────────────────────────

def _search_children(agent: AgentConfig, query_vec: list[float]) -> list[dict]:
    resp = _supabase().rpc("match_documents", {
        "query_embedding": query_vec,
        "match_table":     agent.table_name,
        "match_threshold": agent.similarity_threshold,
        "match_count":     agent.k,
    }).execute()
    return resp.data or []


def _fetch_parents(table_name: str, parent_ids: list[str]) -> list[dict]:
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
    q_vec    = _embeddings().embed_query(question)
    children = _search_children(agent, q_vec)

    if not children:
        logger.debug(f"RAG: 0 resultados para '{question[:60]}'")
        return [], []

    new_children  = [c for c in children if c.get("chunk_level") == "child"]
    flat_children = [c for c in children if c.get("chunk_level") != "child"]

    context_chunks: list[dict] = []

    if new_children:
        seen_pids: set[str] = set()
        ordered_pids: list[str] = []
        for c in new_children:
            pid = c.get("parent_id", "")
            if pid and pid not in seen_pids:
                seen_pids.add(pid)
                ordered_pids.append(pid)

        parents = _fetch_parents(agent.table_name, ordered_pids)
        pid_order = {pid: i for i, pid in enumerate(ordered_pids)}
        parents.sort(key=lambda p: pid_order.get(p.get("parent_id", ""), 999))
        context_chunks.extend(parents)
        logger.debug(
            f"RAG parent-child: {len(new_children)} children → "
            f"{len(parents)} parents para '{question[:60]}'"
        )

    if flat_children:
        context_chunks.extend(flat_children)
        logger.debug(f"RAG flat (legado): {len(flat_children)} chunks para '{question[:60]}'")

    return context_chunks, children


# ── Montagem do bloco de contexto ─────────────────────────────────────────────

def _context_block(chunks: list[dict]) -> str:
    if not chunks:
        return "Nenhum documento relevante encontrado na base de conhecimento."

    parts = []
    for i, c in enumerate(chunks, 1):
        meta = c.get("metadata") or {}
        src  = meta.get("file_name") or c.get("file_name", "desconhecido")
        sec  = " > ".join(filter(None, [meta.get("h1", ""), meta.get("h2", "")]))
        hdr  = f"[{i}] {src}" + (f" | {sec}" if sec else "")
        if c.get("chunk_level") == "parent":
            hdr += " [contexto completo]"
        parts.append(f"{hdr}\n{c['content']}")

    return "\n\n---\n\n".join(parts)


def _build_sources(children: list[dict]) -> list[dict]:
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
        return {"answer": f"Agente '{agent_id}' não encontrado.", "sources": [], "agent": agent_id}

    # ── Busca híbrida: NCM exata + semântica ──────────────────────────────────
    ncm_codes   = _extract_ncms_from_question(question)
    ncm_results = _search_ncm(ncm_codes) if ncm_codes else []

    if ncm_codes:
        logger.info(f"  🔍 NCMs detectados na pergunta: {ncm_codes} → {len(ncm_results)} registros")

    # Busca semântica normal
    context_chunks, children = _retrieve(agent, question)

    # ── Monta contexto combinado ──────────────────────────────────────────────
    ncm_ctx      = _ncm_context_block(ncm_results)
    semantic_ctx = _context_block(context_chunks)

    # NCM vai primeiro — é contexto exato e estruturado
    if ncm_ctx:
        full_context = f"{ncm_ctx}\n\n{'═'*50}\n\n📄 CONTEXTO DOCUMENTAL:\n\n{semantic_ctx}"
    else:
        full_context = semantic_ctx

    system = f"""{agent.system_prompt}

═══════════════════════════════════════
BASE DE CONHECIMENTO (use como referência principal):

{full_context}
═══════════════════════════════════════

INSTRUÇÕES IMPORTANTES:
- Quando houver dados de NCM específicos, priorize-os na resposta
- Cite sempre o dispositivo legal (artigo, convênio, protocolo)
- Se o NCM não estiver na base, informe claramente
- Responda com base no contexto acima"""

    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

    messages = [SystemMessage(content=system)]
    for msg in (history or []):
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=question))

    response = await _get_llm().ainvoke(messages)

    # Inclui fontes NCM nas sources
    ncm_sources = [
        {
            "file":    r.get("file_name", ""),
            "section": f"NCM {r.get('ncm')} — {r.get('beneficio', '')}",
            "score":   1.0,  # busca exata = score máximo
        }
        for r in ncm_results[:5]
        if r.get("file_name")
    ]

    return {
        "answer":      response.content,
        "sources":     ncm_sources + _build_sources(children),
        "agent":       agent.name,
        "agent_id":    agent_id,
        "llm":         _get_llm_name(),
        "ncm_found":   [r.get("ncm") for r in ncm_results],
    }
