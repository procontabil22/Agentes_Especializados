"""
pipeline.py — Pipeline em Duas Fases

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FASE 1 — process_pdf()
  PDF (Drive) → Docling → Chunks → LLM extrai JSON estruturado
  → Salva arquivo .json no Google Drive (mesma pasta do PDF)
  → Retorna status "json_saved"

FASE 2 — index_from_json()
  Lê .json do Google Drive → Gera embeddings (OpenAI)
  → Upsert no Supabase (tabela vetorizada do agente)
  → Retorna status "indexed"

VANTAGENS DA ARQUITETURA EM DUAS FASES:
  ✓ JSON auditável e editável antes de indexar
  ✓ Reindexação barata (sem re-rodar Docling + LLM)
  ✓ Rastreabilidade: cada PDF tem um .json correspondente no Drive
  ✓ Custo: Docling (CPU) e LLM (tokens) executam só uma vez

ESTRUTURA DO JSON NO DRIVE:
  analista_fiscal/
    ├── Lei_7689_1988_CSLL.pdf
    └── Lei_7689_1988_CSLL.json   ← gerado pela Fase 1

ESTRUTURA DO JSON:
  {
    "file_name": "Lei_7689_1988_CSLL.pdf",
    "file_hash": "sha256...",
    "doc_type": "legislacao",
    "pages": 8,
    "indexed_at": "2026-...",
    "chunks": [
      {
        "parent_id": "uuid",
        "chunk_level": "parent",
        "chunk_index": 0,
        "unit_type": "artigo",
        "unit_number": "1",
        "content": "Art. 1º ...",
        "h1": "...", "h2": "...",
        "structured": { ... JSON do LLM ... },
        "children": [
          { "chunk_index": 0, "content": "...", "parent_id": "uuid" }
        ]
      }
    ]
  }

CHUNKING INTELIGENTE (adaptativo por tipo):
  LEGISLAÇÃO/TRABALHISTA/SOCIETÁRIO → por ARTIGO
  CONVÊNIO CONFAZ                  → por CLÁUSULA
  NORMA TÉCNICA NBC/CPC            → por SEÇÃO (H1/H2/H3)
  GENÉRICO                         → seções + fallback artigos

EXTRAÇÃO JSON (LLM por tipo de documento):
  legislacao   → tipo_norma, artigo, beneficio_fiscal, tributo, ncm_cfop
  convenio     → numero_convenio, clausula, tipo, estados, aliquota_mva
  norma_tecnica→ norma, tipo_orientacao, metodo_criterio
  trabalhista  → tipo, beneficiario, prazo_valor, evento_esocial
  societario   → tipo_societario, fase_ciclo, orgao_registro
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import hashlib
import json
import re
import uuid
from datetime import datetime
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from loguru import logger

from settings import settings


# ══════════════════════════════════════════════════════════════════════════════
# TIPOS DE DOCUMENTO
# ══════════════════════════════════════════════════════════════════════════════

class DocType(str, Enum):
    LEGISLACAO    = "legislacao"
    CONVENIO      = "convenio"
    NORMA_TECNICA = "norma_tecnica"
    TRABALHISTA   = "trabalhista"
    SOCIETARIO    = "societario"
    GENERICO      = "generico"


_FOLDER_DOCTYPE: dict[str, DocType] = {
    "analista_fiscal":               DocType.LEGISLACAO,
    "analista_contabil":             DocType.NORMA_TECNICA,
    "analista_departamento_pessoal": DocType.TRABALHISTA,
    "analista_societario":           DocType.SOCIETARIO,
    "analista_abertura_empresas":    DocType.SOCIETARIO,
}

_RE_CONVENIO = re.compile(r"conv[eê]nio\s+icms|protocolo\s+icms|ajuste\s+sinief|confaz", re.I)
_RE_NORMA    = re.compile(r"\bnbc\s+t[ga]\b|\bcpc\s+\d|\bcfc\b|\bifrs\b|\bicpc\b|\bocpc\b", re.I)
_RE_ARTIGO   = re.compile(r"(?m)^\s*(?:Art(?:igo)?\.?\s*\d+[º°oa]?|§\s*\d+[º°oa]?)\s*[.\-–—]")
_RE_CLAUSULA = re.compile(r"(?mi)^\s*Cl[aá]usula\s+\w+")
_RE_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


# ══════════════════════════════════════════════════════════════════════════════
# PROMPTS DE EXTRAÇÃO JSON POR TIPO
# ══════════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPTS: dict[DocType, str] = {

    DocType.LEGISLACAO: """\
Você é especialista em legislação tributária brasileira.
Analise o trecho e retorne SOMENTE um objeto JSON válido, sem markdown:
{
  "tipo_norma": "lei|decreto|regulamento|instrucao_normativa|portaria|resolucao|emenda",
  "numero_norma": "ex: 7.799/2002 ou null",
  "artigo": "ex: Art. 4º ou null",
  "inciso": "ex: II ou null",
  "paragrafo": "ex: § 3º ou null",
  "assunto": "resumo objetivo em até 12 palavras",
  "beneficio_fiscal": {
    "tipo": "isencao|reducao_bc|diferimento|credito_outorgado|st|imunidade|nenhum",
    "produto_operacao": "produto/serviço beneficiado ou null",
    "percentual": "ex: 100% ou null",
    "condicao": "condição para fruição ou null",
    "vigencia": "indeterminado|dd/mm/aaaa|null"
  },
  "tributo": "ICMS|IPI|PIS|COFINS|IRPJ|CSLL|ISS|IOF|todos|null",
  "uf_aplicacao": "MA|todos|null",
  "ncm_cfop": "código NCM ou CFOP se mencionado ou null",
  "palavras_chave": ["até 6 termos relevantes"]
}""",

    DocType.CONVENIO: """\
Você é especialista em convênios CONFAZ, protocolos e ajustes SINIEF.
Retorne SOMENTE um objeto JSON válido:
{
  "numero_convenio": "ex: Convênio ICMS 142/2018 ou null",
  "clausula": "ex: Cláusula 3ª ou null",
  "assunto": "resumo objetivo em até 12 palavras",
  "tipo": "isencao|st|reducao|diferimento|obrigacao_acessoria|credenciamento|nfe|mdf_e|outro",
  "estados_signatarios": ["MA","PA"] ou "todos" ou null,
  "produto_operacao": "produto/operação ou null",
  "condicao": "condição principal ou null",
  "aliquota_mva": "ex: 35% MVA ou 12% ou null",
  "ncm": "código NCM se mencionado ou null",
  "palavras_chave": ["até 6 termos"]
}""",

    DocType.NORMA_TECNICA: """\
Você é especialista em normas contábeis brasileiras (NBC TG, CPC, ITG, IFRS).
Retorne SOMENTE um objeto JSON válido:
{
  "norma": "ex: NBC TG 26 / CPC 26 ou null",
  "item_paragrafo": "número do item ou parágrafo ou null",
  "assunto": "resumo objetivo em até 12 palavras",
  "tipo_orientacao": "objetivo|alcance|reconhecimento|mensuracao|divulgacao|apresentacao|definicao|transicao",
  "aplica_se_a": "PME|grande|entidade_sem_fins|todas ou null",
  "metodo_criterio": "método ou critério contábil principal ou null",
  "conta_elemento": "nome da conta ou elemento patrimonial ou null",
  "vigencia": "data de vigência ou null",
  "palavras_chave": ["até 6 termos"]
}""",

    DocType.TRABALHISTA: """\
Você é especialista em direito do trabalho, previdência e eSocial.
Retorne SOMENTE um objeto JSON válido:
{
  "tipo_norma": "clt|lei|decreto|nr|instrucao_normativa|portaria|esocial",
  "artigo": "ex: Art. 7º ou null",
  "assunto": "resumo objetivo em até 12 palavras",
  "tipo": "direito_empregado|obrigacao_empregador|beneficio_prev|seguranca_trabalho|rescisao|ferias|salario|fgts|inss|esocial",
  "beneficiario": "empregado|empregador|autonomo|mei|todos",
  "prazo_valor": "prazo ou valor de referência ou null",
  "evento_esocial": "código S-xxxx se mencionado ou null",
  "condicao": "condição de aplicação ou null",
  "palavras_chave": ["até 6 termos"]
}""",

    DocType.SOCIETARIO: """\
Você é especialista em direito empresarial e registros mercantis.
Retorne SOMENTE um objeto JSON válido:
{
  "tipo_norma": "codigo_civil|lei|instrucao_normativa_drei|resolucao|decreto",
  "artigo": "ex: Art. 1.052 ou null",
  "assunto": "resumo objetivo em até 12 palavras",
  "tipo_societario": "LTDA|SA|SLU|MEI|EIRELI|cooperativa|EI|todos|nenhum",
  "fase_ciclo": "constituicao|alteracao|dissolucao|liquidacao|registro|transformacao|fusao|cisao|geral",
  "obrigacao_direito": "descrição da obrigação ou direito ou null",
  "orgao_registro": "JUCEMA|DREI|RFB|cartorio|municipio|null",
  "prazo_valor": "prazo legal ou capital mínimo ou null",
  "palavras_chave": ["até 6 termos"]
}""",

    DocType.GENERICO: """\
Analise o trecho e retorne SOMENTE um objeto JSON válido:
{
  "assunto": "resumo objetivo em até 12 palavras",
  "tipo_conteudo": "definicao|regra|procedimento|tabela|exemplo|outro",
  "normas_citadas": ["leis ou normas mencionadas"],
  "entidades": ["organizações ou órgãos mencionados"],
  "palavras_chave": ["até 6 termos"]
}""",
}


# ══════════════════════════════════════════════════════════════════════════════
# TAMANHOS DE CHUNKS
# ══════════════════════════════════════════════════════════════════════════════

_PARENT_SIZE    = settings.CHUNK_SIZE * 12
_PARENT_OVERLAP = 100
_CHILD_SIZE     = settings.CHUNK_SIZE * 2
_CHILD_OVERLAP  = settings.CHUNK_OVERLAP * 2
_ARTICLE_MAX    = settings.CHUNK_SIZE * 8


# ══════════════════════════════════════════════════════════════════════════════
# CLIENTES LAZY
# ══════════════════════════════════════════════════════════════════════════════

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


def _make_llm(provider: str):
    """Instancia um LLM pelo nome do provider."""
    if provider == "anthropic" and settings.ANTHROPIC_API_KEY:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            anthropic_api_key=settings.ANTHROPIC_API_KEY,
            max_tokens=600,
            temperature=0,
        )
    if provider == "openai" and settings.OPENAI_API_KEY:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=settings.OPENAI_API_KEY,
            max_tokens=600,
            temperature=0,
        )
    if provider == "gemini" and settings.GEMINI_API_KEY:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=settings.GEMINI_API_KEY,
            max_output_tokens=600,
        )
    return None


def _get_llm_with_fallback():
    """
    Retorna o primeiro LLM disponível e com créditos.
    Testa cada provider em ordem e faz fallback automático se falhar.
    Ordem: Anthropic → OpenAI → Gemini
    """
    providers = []
    if settings.ANTHROPIC_API_KEY:
        providers.append("anthropic")
    if settings.OPENAI_API_KEY:
        providers.append("openai")
    if settings.GEMINI_API_KEY:
        providers.append("gemini")

    if not providers:
        raise RuntimeError("Nenhuma chave de LLM configurada.")

    return providers  # retorna lista ordenada para tentar em sequência


@lru_cache(maxsize=1)
def _converter():
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat
    opts = PdfPipelineOptions()
    opts.do_ocr = True
    opts.do_table_structure = True
    opts.table_structure_options.do_cell_matching = True
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )


# ══════════════════════════════════════════════════════════════════════════════
# DETECÇÃO DE TIPO
# ══════════════════════════════════════════════════════════════════════════════

def _detect_doc_type(folder_name: str, filename: str, sample: str) -> DocType:
    base  = _FOLDER_DOCTYPE.get(folder_name, DocType.GENERICO)
    probe = (filename + " " + sample[:800]).lower()
    if _RE_CONVENIO.search(probe):
        return DocType.CONVENIO
    if _RE_NORMA.search(probe):
        return DocType.NORMA_TECNICA
    return base


# ══════════════════════════════════════════════════════════════════════════════
# EXTRAÇÃO JSON VIA LLM
# ══════════════════════════════════════════════════════════════════════════════

def _extract_json(content: str, doc_type: DocType) -> dict:
    """
    Extrai JSON via LLM com fallback automático entre providers.
    Se Anthropic falhar (sem créditos, erro 400/429), tenta OpenAI.
    Se OpenAI falhar, tenta Gemini.
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    system  = _SYSTEM_PROMPTS.get(doc_type, _SYSTEM_PROMPTS[DocType.GENERICO])
    snippet = content[:3000]
    msgs    = [
        SystemMessage(content=system),
        HumanMessage(content=f"TRECHO DO DOCUMENTO:\n\n{snippet}"),
    ]

    providers = _get_llm_with_fallback()
    last_error = None

    for provider in providers:
        llm = _make_llm(provider)
        if not llm:
            continue
        try:
            resp  = llm.invoke(msgs)
            raw   = resp.content.strip()
            fence = _RE_JSON_FENCE.search(raw)
            if fence:
                raw = fence.group(1)
            result = json.loads(raw)
            if provider != providers[0]:
                logger.debug(f"    ↳ Fallback para {provider} funcionou")
            return result
        except json.JSONDecodeError as e:
            return {"assunto": content[:80], "_parse_error": str(e)}
        except Exception as e:
            err_str = str(e)
            # Erros de crédito/autenticação → tenta próximo provider
            if any(k in err_str.lower() for k in [
                "credit", "balance", "quota", "rate_limit",
                "insufficient", "billing", "payment", "403", "429"
            ]):
                logger.debug(f"    ↳ {provider} sem créditos/quota — tentando próximo")
                last_error = err_str
                continue
            # Outros erros → retorna imediatamente
            return {"assunto": content[:80], "_llm_error": err_str}

    return {"assunto": content[:80], "_llm_error": f"Todos os providers falharam: {last_error}"}


# ══════════════════════════════════════════════════════════════════════════════
# DETECÇÃO DE HTML DISFARÇADO DE PDF
# ══════════════════════════════════════════════════════════════════════════════

def _is_html(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            h = f.read(512).lower()
        return b"<!doctype html" in h or b"<html" in h or b"<meta" in h
    except Exception:
        return False


def _ensure_correct_extension(path: Path) -> Path:
    """Renomeia .pdf para .html se o conteúdo for HTML (Planalto retorna HTML)."""
    if not _is_html(path):
        return path
    html_path = path.with_suffix(".html")
    import shutil
    shutil.copy2(path, html_path)
    logger.info("  📄 HTML detectado — usando backend HTML do Docling")
    return html_path


# ══════════════════════════════════════════════════════════════════════════════
# CHUNKING — UNIDADES LEGAIS (Artigos / Cláusulas)
# ══════════════════════════════════════════════════════════════════════════════

def _split_by_legal_unit(
    markdown: str,
    source_meta: dict,
    pattern: re.Pattern,
    unit_name: str,
) -> tuple[list[dict], list[dict]]:
    boundaries = [m.start() for m in pattern.finditer(markdown)]
    if not boundaries:
        return _split_by_sections(markdown, source_meta)
    boundaries.append(len(markdown))

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHILD_SIZE, chunk_overlap=_CHILD_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    sub_splitter = RecursiveCharacterTextSplitter(
        chunk_size=_PARENT_SIZE, chunk_overlap=_PARENT_OVERLAP,
        separators=["\n\n", "\n", ". "],
    )

    parents: list[dict] = []
    children: list[dict] = []
    parent_idx = child_idx = 0

    for i in range(len(boundaries) - 1):
        unit_text  = markdown[boundaries[i]: boundaries[i + 1]].strip()
        if not unit_text:
            continue
        first_line = unit_text.split("\n")[0].strip()
        art_m      = re.match(r"(?:Art\.?\s*(\d+[º°oa]?)|Cl[aá]usula\s+(\w+))", first_line, re.I)
        unit_num   = (art_m.group(1) or art_m.group(2)) if art_m else str(i + 1)

        base = {**source_meta, "unit_type": unit_name,
                "unit_number": unit_num, "unit_title": first_line[:200]}

        def _mk_parent(text: str) -> str:
            nonlocal parent_idx
            pid = str(uuid.uuid4())
            parents.append({
                "content": text, "parent_id": pid,
                "chunk_level": "parent", "chunk_index": parent_idx,
                "h1": base.get("unit_title", "")[:100], "h2": unit_name,
                "metadata": {**base, "chunk_index": parent_idx,
                              "chunk_level": "parent", "parent_id": pid},
            })
            parent_idx += 1
            return pid

        def _mk_children(text: str, pid: str) -> None:
            nonlocal child_idx
            for ct in child_splitter.split_text(text):
                if not ct.strip():
                    continue
                children.append({
                    "content": ct, "parent_id": pid,
                    "chunk_level": "child", "chunk_index": child_idx,
                    "metadata": {**base, "chunk_index": child_idx,
                                 "chunk_level": "child", "parent_id": pid},
                })
                child_idx += 1

        if len(unit_text) > _ARTICLE_MAX:
            for sub in sub_splitter.split_text(unit_text):
                pid = _mk_parent(sub)
                _mk_children(sub, pid)
        else:
            pid = _mk_parent(unit_text)
            _mk_children(unit_text, pid)

    return parents, children


# ══════════════════════════════════════════════════════════════════════════════
# CHUNKING — SEÇÕES / CABEÇALHOS
# ══════════════════════════════════════════════════════════════════════════════

def _split_by_sections(markdown: str, source_meta: dict) -> tuple[list[dict], list[dict]]:
    h_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )
    sections = h_splitter.split_text(markdown)

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=_PARENT_SIZE, chunk_overlap=_PARENT_OVERLAP,
        separators=["\n\n\n", "\n\n", "\n", ". "],
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHILD_SIZE, chunk_overlap=_CHILD_OVERLAP,
        separators=["\n\n", "\n", ". "],
    )

    parents:  list[dict] = []
    children: list[dict] = []
    parent_idx = child_idx = 0

    for pdoc in parent_splitter.split_documents(sections):
        pid  = str(uuid.uuid4())
        base = {**source_meta, "unit_type": "secao",
                "h1": pdoc.metadata.get("h1", ""),
                "h2": pdoc.metadata.get("h2", ""),
                "h3": pdoc.metadata.get("h3", "")}
        parents.append({
            "content": pdoc.page_content, "parent_id": pid,
            "chunk_level": "parent", "chunk_index": parent_idx,
            "h1": base["h1"], "h2": base["h2"],
            "metadata": {**base, "chunk_index": parent_idx,
                         "chunk_level": "parent", "parent_id": pid},
        })
        parent_idx += 1

        for ct in child_splitter.split_text(pdoc.page_content):
            if not ct.strip():
                continue
            children.append({
                "content": ct, "parent_id": pid,
                "chunk_level": "child", "chunk_index": child_idx,
                "metadata": {**base, "chunk_index": child_idx,
                             "chunk_level": "child", "parent_id": pid},
            })
            child_idx += 1

    return parents, children


# ══════════════════════════════════════════════════════════════════════════════
# TABELAS DOCLING
# ══════════════════════════════════════════════════════════════════════════════

def _extract_tables(docling_result: Any, source_meta: dict) -> list[dict]:
    table_chunks: list[dict] = []
    try:
        for i, table in enumerate(docling_result.document.tables or []):
            try:
                tmd = (table.export_to_markdown()
                       if hasattr(table, "export_to_markdown") else str(table))
                if not tmd.strip() or len(tmd) < 20:
                    continue
                pid = str(uuid.uuid4())
                idx = 90000 + i
                base = {**source_meta, "unit_type": "tabela",
                        "table_index": i, "is_table": True,
                        "h1": source_meta.get("file_name", ""),
                        "h2": f"Tabela {i + 1}"}
                table_chunks.append({
                    "content": f"[TABELA {i + 1}]\n{tmd}",
                    "parent_id": pid,
                    "chunk_level": "parent", "chunk_index": idx,
                    "h1": base["h1"], "h2": base["h2"],
                    "metadata": {**base, "chunk_index": idx,
                                 "chunk_level": "parent", "parent_id": pid},
                })
            except Exception as e:
                logger.debug(f"  ⚠ Tabela {i}: {e}")
    except Exception as e:
        logger.debug(f"  ⚠ Extração tabelas: {e}")
    return table_chunks


# ══════════════════════════════════════════════════════════════════════════════
# SELEÇÃO DA ESTRATÉGIA DE CHUNKING
# ══════════════════════════════════════════════════════════════════════════════

def _select_strategy(doc_type: DocType, markdown: str, source_meta: dict):
    n_art = len(_RE_ARTIGO.findall(markdown))
    n_cla = len(_RE_CLAUSULA.findall(markdown))
    logger.debug(f"  Artigos: {n_art} | Cláusulas: {n_cla}")

    if doc_type == DocType.CONVENIO or (n_cla > 3 and n_cla >= n_art):
        logger.info(f"  📐 Chunking: CLÁUSULAS ({n_cla})")
        return _split_by_legal_unit(markdown, source_meta, _RE_CLAUSULA, "clausula")
    if doc_type in (DocType.LEGISLACAO, DocType.TRABALHISTA, DocType.SOCIETARIO) and n_art > 3:
        logger.info(f"  📐 Chunking: ARTIGOS ({n_art})")
        return _split_by_legal_unit(markdown, source_meta, _RE_ARTIGO, "artigo")
    if doc_type == DocType.NORMA_TECNICA:
        logger.info("  📐 Chunking: SEÇÕES (norma técnica)")
        return _split_by_sections(markdown, source_meta)
    if n_art > 3:
        logger.info(f"  📐 Chunking: ARTIGOS fallback ({n_art})")
        return _split_by_legal_unit(markdown, source_meta, _RE_ARTIGO, "artigo")
    logger.info("  📐 Chunking: SEÇÕES genérico")
    return _split_by_sections(markdown, source_meta)


# ══════════════════════════════════════════════════════════════════════════════
# UTILITÁRIOS
# ══════════════════════════════════════════════════════════════════════════════

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def _json_already_exists(table: str, file_hash: str) -> bool:
    """Verifica se o JSON já foi gerado (parent com este hash existe no Supabase)."""
    resp = (
        _supabase().table(table)
        .select("id")
        .eq("file_hash", file_hash)
        .eq("chunk_level", "parent")
        .limit(1)
        .execute()
    )
    return len(resp.data) > 0


def _embedding_already_exists(table: str, file_hash: str) -> bool:
    """Verifica se os embeddings já foram gerados para este arquivo."""
    resp = (
        _supabase().table(table)
        .select("id")
        .eq("file_hash", file_hash)
        .eq("chunk_level", "child")
        .not_.is_("embedding", "null")
        .limit(1)
        .execute()
    )
    return len(resp.data) > 0


def _upsert_batch(table: str, rows: list[dict]) -> None:
    for i in range(0, len(rows), 100):
        _supabase().table(table).upsert(
            rows[i: i + 100],
            on_conflict="file_hash,chunk_index,chunk_level",
        ).execute()


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 — process_pdf()
# PDF → Docling → Chunks → LLM JSON → Salva .json no Drive
# ══════════════════════════════════════════════════════════════════════════════

def process_pdf(
    pdf_path: Path,
    file_name: str,
    file_id: str,
    folder_name: str,
    table_name: str,
    modified_at: str = "",
) -> dict[str, Any]:
    """
    Fase 1: Converte PDF em chunks estruturados com JSON do LLM
    e salva o resultado como .json no Google Drive.
    NÃO gera embeddings nem grava no Supabase.
    """
    logger.info(f"▶ [FASE 1] {file_name}")

    file_hash = _sha256(pdf_path)

    # Verifica se o JSON já foi gerado anteriormente
    if _json_already_exists(table_name, file_hash):
        logger.info("  ⏭ JSON já existe no Drive (mesmo hash) — pulando Fase 1")
        return {"status": "json_exists", "file": file_name, "file_hash": file_hash}

    # ── Detecta HTML disfarçado de PDF ────────────────────────────────────────
    doc_path = _ensure_correct_extension(pdf_path)

    # ── Docling: documento → markdown estruturado ─────────────────────────────
    logger.info("  🔍 Docling: convertendo documento...")
    result   = _converter().convert(str(doc_path))
    markdown = result.document.export_to_markdown()
    pages    = len(result.document.pages) if result.document.pages else 0
    logger.info(f"  Docling OK: {pages} páginas | {len(markdown):,} chars")

    # ── Detecta tipo de documento ─────────────────────────────────────────────
    doc_type = _detect_doc_type(folder_name, file_name, markdown)
    logger.info(f"  📄 Tipo: {doc_type.value}")

    source_meta = {
        "file_name":   file_name,
        "file_id":     file_id,
        "file_hash":   file_hash,
        "folder_name": folder_name,
        "page_count":  pages,
        "modified_at": modified_at,
        "indexed_at":  datetime.utcnow().isoformat(),
        "agent":       folder_name,
        "doc_type":    doc_type.value,
    }

    # ── Chunking inteligente ──────────────────────────────────────────────────
    parents, children = _select_strategy(doc_type, markdown, source_meta)

    # ── Tabelas Docling ───────────────────────────────────────────────────────
    table_chunks = _extract_tables(result, source_meta)
    if table_chunks:
        parents.extend(table_chunks)
        logger.info(f"  📊 {len(table_chunks)} tabela(s) extraída(s)")

    logger.info(f"  → {len(parents)} parents | {len(children)} children")

    # ── Extração JSON via LLM (só parents, max 200) ───────────────────────────
    MAX_JSON = 200
    logger.info(f"  🤖 Extraindo JSON ({min(len(parents), MAX_JSON)} parents)...")
    parent_jsons: list[dict] = []
    for i, p in enumerate(parents):
        if i >= MAX_JSON:
            parent_jsons.append({})
            continue
        parent_jsons.append(_extract_json(p["content"], doc_type))
        if (i + 1) % 25 == 0:
            ok = sum(1 for j in parent_jsons if j and "_llm_error" not in j)
            logger.debug(f"    {i + 1}/{min(len(parents), MAX_JSON)} | {ok} OK")

    json_ok = sum(1 for j in parent_jsons if j and "_llm_error" not in j and "_parse_error" not in j)
    logger.info(f"  ✓ JSON: {json_ok}/{len(parents)} sem erro")

    # ── Monta payload JSON para salvar no Drive ───────────────────────────────
    pid_to_json = {p["parent_id"]: parent_jsons[i] for i, p in enumerate(parents)}

    chunks_payload = []
    for i, p in enumerate(parents):
        pjson    = parent_jsons[i]
        p_children = [
            {
                "chunk_index": c["chunk_index"],
                "content":     c["content"],
                "parent_id":   c["parent_id"],
            }
            for c in children if c["parent_id"] == p["parent_id"]
        ]
        chunks_payload.append({
            "parent_id":   p["parent_id"],
            "chunk_index": p["chunk_index"],
            "chunk_level": "parent",
            "unit_type":   p["metadata"].get("unit_type", ""),
            "unit_number": p["metadata"].get("unit_number", ""),
            "unit_title":  p["metadata"].get("unit_title", ""),
            "h1":          p.get("h1", ""),
            "h2":          p.get("h2", ""),
            "content":     p["content"],
            "structured":  pjson,
            "children":    p_children,
        })

    json_payload = {
        "file_name":    file_name,
        "file_id":      file_id,
        "file_hash":    file_hash,
        "folder_name":  folder_name,
        "table_name":   table_name,
        "doc_type":     doc_type.value,
        "pages":        pages,
        "total_parents": len(parents),
        "total_children": len(children),
        "json_ok":      json_ok,
        "generated_at": datetime.utcnow().isoformat(),
        "chunks":       chunks_payload,
    }

    # ── Salva .json no Google Drive ───────────────────────────────────────────
    from gdrive import _get_service, _get_or_create_folder, _upload_bytes_to_drive, _get_file_id_in_folder

    json_filename = file_name.rsplit(".", 1)[0] + ".json"
    json_bytes    = json.dumps(json_payload, ensure_ascii=False, indent=2).encode("utf-8")

    svc       = _get_service()
    folder_id = _get_or_create_folder(svc, folder_name, settings.GDRIVE_ROOT_FOLDER_ID)

    # Remove versão anterior se existir
    old_id = _get_file_id_in_folder(svc, json_filename, folder_id)
    if old_id:
        try:
            svc.files().delete(fileId=old_id, supportsAllDrives=True).execute()
        except Exception:
            pass

    drive_json_id = _upload_bytes_to_drive(
        svc, json_bytes, json_filename, folder_id,
        mime_type="application/json"
    )
    logger.success(f"  ✅ JSON salvo no Drive: {json_filename} → {drive_json_id}")

    return {
        "status":       "json_saved",
        "file":         file_name,
        "json_file":    json_filename,
        "drive_json_id": drive_json_id,
        "file_hash":    file_hash,
        "doc_type":     doc_type.value,
        "parents":      len(parents),
        "children":     len(children),
        "json_ok":      json_ok,
        "pages":        pages,
    }


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 — index_from_json()
# Lê .json do Drive → Embeddings → Upsert no Supabase
# ══════════════════════════════════════════════════════════════════════════════

def index_from_json(
    json_filename: str,
    folder_name: str,
    table_name: str,
) -> dict[str, Any]:
    """
    Fase 2: Lê o .json do Google Drive, gera embeddings nos children
    e faz upsert no Supabase (tabela vetorizada do agente).
    """
    logger.info(f"▶ [FASE 2] {json_filename} → {table_name}")

    from gdrive import _get_service, _get_or_create_folder, download_file_bytes, _get_file_id_in_folder

    svc       = _get_service()
    folder_id = _get_or_create_folder(svc, folder_name, settings.GDRIVE_ROOT_FOLDER_ID)

    # ── Baixa o JSON do Drive ─────────────────────────────────────────────────
    json_file_id = _get_file_id_in_folder(svc, json_filename, folder_id)
    if not json_file_id:
        logger.error(f"  ✗ {json_filename} não encontrado no Drive")
        return {"status": "error", "file": json_filename, "error": "JSON não encontrado no Drive"}

    raw_bytes = download_file_bytes(svc, json_file_id)
    payload   = json.loads(raw_bytes.decode("utf-8"))

    file_hash = payload["file_hash"]
    file_name = payload["file_name"]
    doc_type  = payload.get("doc_type", "generico")

    # ── Verifica se embeddings já existem ────────────────────────────────────
    if _embedding_already_exists(table_name, file_hash):
        logger.info(f"  ⏭ Embeddings já existem para {file_name}")
        return {"status": "already_indexed", "file": file_name}

    chunks    = payload.get("chunks", [])
    source_meta = {
        "file_name":   file_name,
        "file_id":     payload.get("file_id", ""),
        "file_hash":   file_hash,
        "folder_name": folder_name,
        "page_count":  payload.get("pages", 0),
        "modified_at": "",
        "indexed_at":  datetime.utcnow().isoformat(),
        "agent":       folder_name,
        "doc_type":    doc_type,
    }

    # ── Reconstrói parents e children do JSON ────────────────────────────────
    parent_rows: list[dict] = []
    child_rows:  list[dict] = []
    all_child_texts: list[str] = []

    for chunk in chunks:
        pjson = chunk.get("structured", {})
        enriched_meta = {
            **source_meta,
            "unit_type":   chunk.get("unit_type", ""),
            "unit_number": chunk.get("unit_number", ""),
            "unit_title":  chunk.get("unit_title", ""),
            "h1":          chunk.get("h1", ""),
            "h2":          chunk.get("h2", ""),
            "chunk_index": chunk["chunk_index"],
            "chunk_level": "parent",
            "parent_id":   chunk["parent_id"],
            "structured":  pjson,
        }
        if pjson:
            if "assunto"          in pjson: enriched_meta["assunto"]          = pjson["assunto"]
            if "palavras_chave"   in pjson: enriched_meta["palavras_chave"]   = pjson["palavras_chave"]
            if "tributo"          in pjson: enriched_meta["tributo"]          = pjson["tributo"]
            if "beneficio_fiscal" in pjson: enriched_meta["beneficio_fiscal"] = pjson["beneficio_fiscal"]

        parent_rows.append({
            "content":     chunk["content"],
            "metadata":    enriched_meta,
            "file_name":   file_name,
            "file_hash":   file_hash,
            "folder":      folder_name,
            "agent":       folder_name,
            "chunk_index": chunk["chunk_index"],
            "chunk_level": "parent",
            "parent_id":   chunk["parent_id"],
            "h1":          chunk.get("h1", "")[:100],
            "h2":          chunk.get("h2", ""),
            "indexed_at":  source_meta["indexed_at"],
            "embedding":   None,
        })

        for child in chunk.get("children", []):
            child_meta = {**enriched_meta,
                          "chunk_index": child["chunk_index"],
                          "chunk_level": "child",
                          "parent_id":   child["parent_id"]}
            child_rows.append({
                "content":     child["content"],
                "metadata":    child_meta,
                "file_name":   file_name,
                "file_hash":   file_hash,
                "folder":      folder_name,
                "agent":       folder_name,
                "chunk_index": child["chunk_index"],
                "chunk_level": "child",
                "parent_id":   child["parent_id"],
                "h1":          chunk.get("h1", "")[:100],
                "h2":          chunk.get("h2", ""),
                "indexed_at":  source_meta["indexed_at"],
            })
            all_child_texts.append(child["content"])

    logger.info(f"  → {len(parent_rows)} parents | {len(child_rows)} children")

    # ── Gera embeddings nos children (LangChain + OpenAI) ────────────────────
    child_vecs: list[list[float]] = []
    logger.info(f"  🔢 Gerando {len(all_child_texts)} embeddings...")
    for i in range(0, len(all_child_texts), settings.BATCH_SIZE):
        batch = all_child_texts[i: i + settings.BATCH_SIZE]
        child_vecs.extend(_embeddings().embed_documents(batch))
        logger.debug(f"    Embeddings: {min(i + settings.BATCH_SIZE, len(all_child_texts))}/{len(all_child_texts)}")

    for row, vec in zip(child_rows, child_vecs):
        row["embedding"] = vec

    # ── Upsert parents (sem embedding) ───────────────────────────────────────
    _upsert_batch(table_name, parent_rows)
    logger.debug(f"  ✓ {len(parent_rows)} parents gravados")

    # ── Upsert children (com embedding) ──────────────────────────────────────
    _upsert_batch(table_name, child_rows)
    logger.success(f"  ✅ {len(child_rows)} children gravados em '{table_name}'")

    return {
        "status":   "indexed",
        "file":     file_name,
        "table":    table_name,
        "parents":  len(parent_rows),
        "children": len(child_rows),
        "pages":    payload.get("pages", 0),
    }
