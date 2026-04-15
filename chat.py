"""
chat.py — RAG com Parent-Child Retrieval + Resolução Produto→NCM

Fluxo atualizado (v2):
  1. embed(pergunta)
  2. [NOVO] resolve_product_ncms(pergunta) → tenta traduzir nomes de
     produtos como "refrigerante" para NCM antes da busca semântica
  3. _extract_ncms_from_question() → NCMs numéricos explícitos na pergunta
  4. _search_ncm(todos_ncms) → consulta kb_ncm_fiscal com NCMs completos +
     variações (8, 6, 4 dígitos)
  5. match_documents → retorna N children mais similares
     (threshold reduzido para 0.62, k aumentado para 12)
  6. coleta parent_ids únicos dos children
  7. busca conteúdo completo dos parents (contexto rico)
  8. monta prompt → LLM → resposta

Correções aplicadas:
  - CAUSA 1/2: product_ncm.resolve_product_ncms() traduz produto→NCM
  - CAUSA 3: similarity_threshold reduzido de 0.70 → 0.62
  - CAUSA 4: k aumentado de 8 → 12; variações de NCM (6 e 4 dígitos)

Retrocompatibilidade:
  Chunks antigos (chunk_level='flat') são retornados diretamente pelo
  match_documents e usados como contexto (sem lookup de parent).
"""

from functools import lru_cache
import re

from loguru import logger

from agents import AgentConfig, get_agent
from settings import settings
from product_ncm import resolve_product_ncms, get_ncm_variants  # ← NOVO

# Padrão para detectar NCM numérico explícito na pergunta
_RE_NCM_QUERY = re.compile(
    r"\b(\d{4}[.\-]?\d{2}[.\-]?\d{2}[.\-]?\d{2})"
    r"|\b(\d{4}[.\-]?\d{2}[.\-]?\d{2})"
    r"|\b(\d{4}[.\-]?\d{2})"
    r"|\b(\d{4})\b"
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
      1. OpenAI (GPT)  2. Google Gemini  3. Grok (xAI)  4. DeepSeek
    """
    if settings.OPENAI_API_KEY:
        from langchain_openai import ChatOpenAI
        logger.info("LLM selecionado: OpenAI (GPT-4o-mini)")
        return ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=settings.OPENAI_API_KEY,
            max_tokens=2048,
        )
    if settings.GEMINI_API_KEY:
        from langchain_google_genai import ChatGoogleGenerativeAI
        logger.info("LLM selecionado: Google Gemini")
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=settings.GEMINI_API_KEY,
            max_output_tokens=2048,
        )
    if settings.GROK_API_KEY:
        from langchain_openai import ChatOpenAI
        logger.info("LLM selecionado: Grok (xAI)")
        return ChatOpenAI(
            model="grok-beta",
            openai_api_key=settings.GROK_API_KEY,
            openai_api_base="https://api.x.ai/v1",
            max_tokens=2048,
        )
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
        "Configure ao menos uma das variáveis: OPENAI_API_KEY, "
        "GEMINI_API_KEY, GROK_API_KEY ou DEEPSEEK_API_KEY."
    )


def _get_llm_name() -> str:
    if settings.OPENAI_API_KEY:    return "OpenAI (GPT-4o-mini)"
    if settings.GEMINI_API_KEY:    return "Google Gemini"
    if settings.GROK_API_KEY:      return "Grok (xAI)"
    if settings.DEEPSEEK_API_KEY:  return "DeepSeek"
    return "desconhecido"


# ── Resolução NCM (numérica + por nome de produto) ───────────────────────────

def _extract_ncms_from_question(question: str) -> list[str]:
    """Extrai códigos NCM numéricos explícitos na pergunta."""
    found = []
    for m in _RE_NCM_QUERY.finditer(question):
        ncm = next(g for g in m.groups() if g)
        ncm_digits = re.sub(r"[.\-\s]", "", ncm)
        if len(ncm_digits) >= 4:
            found.append(ncm_digits)
    return list(dict.fromkeys(found))


def _collect_all_ncms(question: str) -> tuple[list[str], list[str]]:
    """
    ── NOVO ──
    Combina duas fontes de NCM:
      1. NCMs numéricos explícitos na pergunta (ex.: "NCM 2202.10.00")
      2. NCMs resolvidos a partir de nomes de produtos (ex.: "refrigerante")

    Retorna:
      (ncms_todos, produtos_detectados)

    Para cada NCM encontrado, gera variações (8, 6, 4 dígitos) para ampliar
    o alcance da busca na tabela kb_ncm_fiscal.
    """
    # Fonte 1: NCMs numéricos na pergunta
    explicit_ncms = _extract_ncms_from_question(question)

    # Fonte 2: Resolução por nome de produto
    product_ncms = resolve_product_ncms(question)

    # Detecta os produtos que geraram NCMs (para log)
    from product_ncm import PRODUCT_NCM_MAP, _normalize
    normalized_q = _normalize(question)
    detected_products = [
        k for k in PRODUCT_NCM_MAP
        if k in normalized_q
    ]

    all_ncms_raw = list(dict.fromkeys(explicit_ncms + product_ncms))

    # Expande cada NCM com variações (8 → 6 → 4 dígitos)
    all_ncms_expanded: list[str] = []
    seen: set[str] = set()
    for ncm in all_ncms_raw:
        for variant in get_ncm_variants(ncm):
            if variant not in seen:
                seen.add(variant)
                all_ncms_expanded.append(variant)

    return all_ncms_expanded, detected_products


def _search_ncm(ncm_codes: list[str], limit: int = 15) -> list[dict]:
    """
    Busca registros NCM na tabela kb_ncm_fiscal.
    Aceita lista de NCMs que pode incluir variações (8, 6, 4 dígitos)
    para ampliar a cobertura quando o NCM exato não existe.
    """
    if not ncm_codes:
        return []

    results = []
    seen_keys: set[str] = set()

    for ncm in ncm_codes:
        try:
            resp = _supabase().rpc("search_ncm", {
                "p_ncm":   ncm,
                "p_uf":    "MA",
                "p_limit": limit,
            }).execute()
            if resp.data:
                for row in resp.data:
                    key = f"{row.get('ncm')}_{row.get('beneficio')}"
                    if key not in seen_keys:
                        seen_keys.add(key)
                        results.append(row)
                logger.debug(f"  NCM {ncm}: {len(resp.data)} registros")
        except Exception as e:
            logger.debug(f"  ⚠ Busca NCM {ncm}: {e}")

        # Também tenta busca por descrição quando o NCM é de capítulo (4 dígitos)
        # ou posição (6 dígitos) — pode não ter exatamente na tabela
        if len(ncm) <= 6:
            try:
                resp2 = _supabase().rpc("search_ncm_by_description", {
                    "p_ncm_prefix": ncm,
                    "p_uf":         "MA",
                    "p_limit":      limit,
                }).execute()
                if resp2.data:
                    for row in resp2.data:
                        key = f"{row.get('ncm')}_{row.get('beneficio')}"
                        if key not in seen_keys:
                            seen_keys.add(key)
                            results.append(row)
                    logger.debug(f"  NCM prefixo {ncm}: {len(resp2.data)} registros adicionais")
            except Exception as e:
                # Função pode não existir ainda — ignora silenciosamente
                logger.debug(f"  ⚠ search_ncm_by_description {ncm}: {e}")

    return results


def _ncm_context_block(ncm_results: list[dict], detected_products: list[str] = None) -> str:
    if not ncm_results:
        return ""

    header = "📦 TRATAMENTO TRIBUTÁRIO POR NCM (RICMS-MA e CONFAZ):\n"
    if detected_products:
        header += f"   Produtos detectados na pergunta: {', '.join(detected_products)}\n"

    parts = [header]
    seen: set[str] = set()

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

    # ── Resolução NCM: numérica + por nome de produto ─────────────────────────
    all_ncm_codes, detected_products = _collect_all_ncms(question)

    if all_ncm_codes:
        logger.info(
            f"  🔍 NCMs resolvidos: {all_ncm_codes} "
            f"(produtos detectados: {detected_products or 'via NCM explícito'})"
        )

    ncm_results = _search_ncm(all_ncm_codes) if all_ncm_codes else []

    if ncm_results:
        logger.info(f"  ✅ kb_ncm_fiscal: {len(ncm_results)} registros encontrados")
    elif all_ncm_codes:
        logger.info(f"  ℹ️  kb_ncm_fiscal: sem registros para {all_ncm_codes}")

    # ── Busca semântica ───────────────────────────────────────────────────────
    context_chunks, children = _retrieve(agent, question)

    # ── Monta contexto combinado ──────────────────────────────────────────────
    ncm_ctx      = _ncm_context_block(ncm_results, detected_products)
    semantic_ctx = _context_block(context_chunks)

    if ncm_ctx:
        full_context = (
            f"{ncm_ctx}\n\n{'═'*50}\n\n"
            f"📄 CONTEXTO DOCUMENTAL (artigos, convênios, regulamentos):\n\n"
            f"{semantic_ctx}"
        )
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

    ncm_sources = [
        {
            "file":    r.get("file_name", ""),
            "section": f"NCM {r.get('ncm')} — {r.get('beneficio', '')}",
            "score":   1.0,
        }
        for r in ncm_results[:5]
        if r.get("file_name")
    ]

    return {
        "answer":             response.content,
        "sources":            ncm_sources + _build_sources(children),
        "agent":              agent.name,
        "agent_id":           agent_id,
        "llm":                _get_llm_name(),
        "ncm_found":          [r.get("ncm") for r in ncm_results],
        "products_detected":  detected_products,
    }
