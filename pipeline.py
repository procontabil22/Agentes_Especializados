"""
pipeline_v2.py — Pipeline em Duas Fases (versão corrigida)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FASE 1 — process_pdf()
  PDF (Drive) → Docling → Chunks → LLM extrai JSON estruturado
  → Salva arquivo .json no Google Drive (mesma pasta do PDF)
  → Retorna status "json_saved"

FASE 2 — index_from_json()
  Lê .json do Google Drive → Gera embeddings (OpenAI)
  → Upsert incremental no Supabase (tabela vetorizada do agente)
  → Retorna status "indexed"

CORREÇÕES v2:
  ✓ [1] Verificação de re-processamento no Drive (não no Supabase)
  ✓ [2] Extração LLM paralela com ThreadPoolExecutor
  ✓ [3] Reutiliza cliente Supabase cacheado em _upsert_ncm_records
  ✓ [4] Recodificação HTML cp1252/latin-1 → UTF-8 explícita
  ✓ [5] Log claro de parents descartados pelo limite MAX_JSON
  ✓ [6] Amostra de detecção de tipo ampliada para 2.000 chars
  ✓ [7] _FOLDER_DOCTYPE carregado de settings (com fallback hardcoded)
  ✓ [8] Prompts LLM carregados de arquivos .txt externos (com fallback inline)
  ✓ [9] Retry com backoff exponencial nas chamadas LLM e embedding
  ✓ [10] Embeddings gravados incrementalmente (children por lote persistido)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import hashlib
import json
import re
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

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


# FIX #7 — carrega mapeamento pasta→DocType de settings se disponível,
# com fallback para o dicionário hardcoded original.
_FOLDER_DOCTYPE_DEFAULT: dict[str, DocType] = {
    "analista_fiscal":               DocType.LEGISLACAO,
    "analista_contabil":             DocType.NORMA_TECNICA,
    "analista_departamento_pessoal": DocType.TRABALHISTA,
    "analista_societario":           DocType.SOCIETARIO,
    "analista_abertura_empresas":    DocType.SOCIETARIO,
}

def _load_folder_doctype() -> dict[str, DocType]:
    raw: dict = getattr(settings, "AGENT_DOCTYPE_MAP", {})
    if not raw:
        return _FOLDER_DOCTYPE_DEFAULT
    result = {}
    for folder, dtype_str in raw.items():
        try:
            result[folder] = DocType(dtype_str)
        except ValueError:
            logger.warning(f"  ⚠ AGENT_DOCTYPE_MAP: tipo desconhecido '{dtype_str}' para '{folder}' — usando GENERICO")
            result[folder] = DocType.GENERICO
    return result

_FOLDER_DOCTYPE = _load_folder_doctype()


_RE_CONVENIO   = re.compile(r"conv[eê]nio\s+icms|protocolo\s+icms|ajuste\s+sinief|confaz", re.I)
_RE_NORMA      = re.compile(r"\bnbc\s+t[ga]\b|\bcpc\s+\d|\bcfc\b|\bifrs\b|\bicpc\b|\bocpc\b", re.I)
_RE_ARTIGO     = re.compile(r"(?m)^\s*(?:Art(?:igo)?\.?\s*\d+[º°oa]?|§\s*\d+[º°oa]?)\s*[.\-–—]")
_RE_CLAUSULA   = re.compile(r"(?mi)^\s*Cl[aá]usula\s+\w+")
_RE_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


# ══════════════════════════════════════════════════════════════════════════════
# FIX #8 — PROMPTS LLM: carregados de arquivos externos, com fallback inline
# Estrutura esperada (opcional): prompts/legislacao.txt, prompts/convenio.txt, ...
# ══════════════════════════════════════════════════════════════════════════════

_PROMPTS_DIR = Path(__file__).parent / "prompts"

_SYSTEM_PROMPTS_INLINE: dict[DocType, str] = {

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


def _load_system_prompts() -> dict[DocType, str]:
    """
    FIX #8: tenta carregar prompts de arquivos .txt em prompts/<doctype>.txt.
    Se o arquivo não existir, usa o prompt inline como fallback.
    Isso permite customizar prompts por analista sem editar o código.
    """
    prompts = dict(_SYSTEM_PROMPTS_INLINE)
    if not _PROMPTS_DIR.exists():
        return prompts
    for doc_type in DocType:
        prompt_file = _PROMPTS_DIR / f"{doc_type.value}.txt"
        if prompt_file.exists():
            try:
                prompts[doc_type] = prompt_file.read_text(encoding="utf-8").strip()
                logger.debug(f"  📝 Prompt externo carregado: {prompt_file.name}")
            except Exception as e:
                logger.warning(f"  ⚠ Falha ao carregar {prompt_file}: {e} — usando inline")
    return prompts

_SYSTEM_PROMPTS = _load_system_prompts()


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


@lru_cache(maxsize=1)
def _llm():
    """LLM para extração de metadados JSON — temperatura 0."""
    if settings.ANTHROPIC_API_KEY:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            anthropic_api_key=settings.ANTHROPIC_API_KEY,
            max_tokens=600,
            temperature=0,
        )
    if settings.OPENAI_API_KEY:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=settings.OPENAI_API_KEY,
            max_tokens=600,
            temperature=0,
        )
    if settings.GEMINI_API_KEY:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=settings.GEMINI_API_KEY,
            max_output_tokens=600,
        )
    raise RuntimeError("Nenhuma chave de LLM configurada.")


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
# FIX #6 — amostra ampliada de 800 para 2.000 chars
# ══════════════════════════════════════════════════════════════════════════════

def _detect_doc_type(folder_name: str, filename: str, sample: str) -> DocType:
    base  = _FOLDER_DOCTYPE.get(folder_name, DocType.GENERICO)
    # FIX #6: usa 2.000 chars para capturar documentos com preâmbulo longo
    probe = (filename + " " + sample[:2000]).lower()
    if _RE_CONVENIO.search(probe):
        return DocType.CONVENIO
    if _RE_NORMA.search(probe):
        return DocType.NORMA_TECNICA
    return base


# ══════════════════════════════════════════════════════════════════════════════
# FIX #9 — EXTRAÇÃO JSON VIA LLM com retry e backoff
# FIX #2 — execução paralela via ThreadPoolExecutor
# ══════════════════════════════════════════════════════════════════════════════

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=15),
    retry=retry_if_exception_type(Exception),
    reraise=False,
)
def _extract_json(content: str, doc_type: DocType) -> dict:
    from langchain_core.messages import HumanMessage, SystemMessage
    system  = _SYSTEM_PROMPTS.get(doc_type, _SYSTEM_PROMPTS[DocType.GENERICO])
    snippet = content[:3000]
    resp = _llm().invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"TRECHO DO DOCUMENTO:\n\n{snippet}"),
    ])
    raw   = resp.content.strip()
    fence = _RE_JSON_FENCE.search(raw)
    if fence:
        raw = fence.group(1)
    return json.loads(raw)


def _extract_json_safe(content: str, doc_type: DocType) -> dict:
    """Wrapper com captura de exceção para uso no executor paralelo."""
    try:
        return _extract_json(content, doc_type)
    except json.JSONDecodeError as e:
        return {"assunto": content[:80], "_parse_error": str(e)}
    except Exception as e:
        return {"assunto": content[:80], "_llm_error": str(e)}


def _extract_json_parallel(
    parents: list[dict],
    doc_type: DocType,
    max_json: int = 200,
    max_workers: int = 8,
) -> list[dict]:
    """
    FIX #2: extrai JSON dos parents em paralelo.
    Parents além de max_json recebem {} (sem custo de LLM).
    FIX #5: loga claramente quantos parents foram descartados.
    """
    total     = len(parents)
    to_process = min(total, max_json)
    skipped    = total - to_process

    if skipped > 0:
        logger.warning(
            f"  ⚠ {skipped} parents acima do limite MAX_JSON={max_json} "
            f"— serão indexados SEM extração JSON estruturado"
        )

    logger.info(f"  🤖 Extraindo JSON ({to_process} parents, {max_workers} workers)...")

    results: list[dict] = [{}] * total

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_idx = {
            ex.submit(_extract_json_safe, parents[i]["content"], doc_type): i
            for i in range(to_process)
        }
        completed = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
            completed += 1
            if completed % 25 == 0 or completed == to_process:
                ok = sum(
                    1 for r in results[:completed]
                    if r and "_llm_error" not in r and "_parse_error" not in r
                )
                logger.debug(f"    {completed}/{to_process} | {ok} OK")

    json_ok = sum(
        1 for r in results[:to_process]
        if r and "_llm_error" not in r and "_parse_error" not in r
    )
    logger.info(f"  ✓ JSON: {json_ok}/{to_process} sem erro (+ {skipped} sem extração)")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# DETECÇÃO DE HTML DISFARÇADO DE PDF
# FIX #4 — recodificação cp1252/latin-1 → UTF-8 explícita
# ══════════════════════════════════════════════════════════════════════════════

def _is_html(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            h = f.read(512).lower()
        return b"<!doctype html" in h or b"<html" in h or b"<meta" in h
    except Exception:
        return False


def _ensure_correct_extension(path: Path) -> Path:
    """
    FIX #4: renomeia .pdf para .html se o conteúdo for HTML,
    recodificando explicitamente de cp1252/latin-1 para UTF-8.
    O Planalto frequentemente serve HTML encodado em cp1252.
    """
    if not _is_html(path):
        return path

    html_path = path.with_suffix(".html")
    raw       = path.read_bytes()

    for enc in ("utf-8", "cp1252", "cp1250", "latin-1"):
        try:
            text = raw.decode(enc)
            html_path.write_text(text, encoding="utf-8")
            logger.info(f"  📄 HTML detectado ({enc} → UTF-8) — usando backend HTML do Docling")
            return html_path
        except UnicodeDecodeError:
            continue

    # Último recurso: copia sem recodificação
    shutil.copy2(path, html_path)
    logger.warning("  📄 HTML detectado (encoding desconhecido) — copiado sem recodificação")
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

_RE_NCM = re.compile(
    r"\b(\d{4})\s*[.\-]?\s*(\d{2})\s*[.\-]?\s*(\d{2})\s*[.\-]?\s*(\d{2})\b"
    r"|\b(\d{4})\s*[.\-]?\s*(\d{2})\s*[.\-]?\s*(\d{2})\b"
    r"|\b(\d{4})\s*[.\-]?\s*(\d{2})\b"
    r"|\b(\d{4})\b"
)
_RE_NCM_BENEFICIO = re.compile(
    r"(isen[çc][aã]o|redu[çc][aã]o|diferimento|suspens[aã]o|"
    r"substitui[çc][aã]o\s+tribut[aá]ria|ST|cr[eé]dito\s+outorgado|"
    r"imunidade|n[aã]o\s+incid[eê]ncia)",
    re.IGNORECASE
)
_RE_PERCENTUAL = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*%|MVA\s+(\d+(?:[.,]\d+)?)\s*%|"
    r"redu[çc][aã]o\s+de\s+(\d+(?:[.,]\d+)?)\s*%",
    re.IGNORECASE
)


def _normalize_ncm(ncm_raw: str) -> str:
    return re.sub(r"[.\-\s]", "", ncm_raw)


def _format_ncm(ncm_norm: str) -> str:
    n = ncm_norm.zfill(8)
    if len(n) == 8:
        return f"{n[:4]}.{n[4:6]}.{n[6:8]}"
    if len(n) == 7:
        return f"{n[:4]}.{n[4:6]}.{n[6:]}"
    return ncm_norm


def _extract_ncms_from_table(table_md: str, source_meta: dict, parent_id: str) -> list[dict]:
    ncm_records: list[dict] = []
    lines = table_md.split("\n")

    col_indices = {"ncm": -1, "descricao": -1, "beneficio": -1,
                   "percentual": -1, "condicao": -1, "dispositivo": -1}

    for line in lines:
        if "|" not in line:
            continue
        cols = [c.strip().lower() for c in line.split("|") if c.strip()]
        if not cols:
            continue

        if any(k in " ".join(cols) for k in ["ncm", "código", "produto", "mercadoria"]):
            for i, col in enumerate(cols):
                if any(k in col for k in ["ncm", "código", "cod"]):
                    col_indices["ncm"] = i
                elif any(k in col for k in ["descri", "produto", "mercadoria", "item"]):
                    col_indices["descricao"] = i
                elif any(k in col for k in ["benefício", "beneficio", "tratamento", "situação"]):
                    col_indices["beneficio"] = i
                elif any(k in col for k in ["%", "alíquota", "aliquota", "percentual", "mva"]):
                    col_indices["percentual"] = i
                elif any(k in col for k in ["condição", "condicao", "requisito"]):
                    col_indices["condicao"] = i
                elif any(k in col for k in ["dispositiv", "base legal", "fundamento", "art"]):
                    col_indices["dispositivo"] = i
            continue

        if re.match(r"^\s*[\|:\-\s]+$", line):
            continue

        raw_cols = [c.strip() for c in line.split("|") if c.strip()]
        if not raw_cols:
            continue

        ncm_raw = ""
        if col_indices["ncm"] >= 0 and col_indices["ncm"] < len(raw_cols):
            ncm_raw = raw_cols[col_indices["ncm"]]
        else:
            for col in raw_cols:
                m = _RE_NCM.search(col)
                if m:
                    ncm_raw = m.group(0)
                    break

        if not ncm_raw:
            continue

        ncm_digits = re.sub(r"[^\d]", "", ncm_raw)
        if len(ncm_digits) < 4:
            continue

        ncm_norm = _normalize_ncm(ncm_raw)
        ncm_fmt  = _format_ncm(ncm_norm)
        full_line = " ".join(raw_cols)

        descricao = ""
        if col_indices["descricao"] >= 0 and col_indices["descricao"] < len(raw_cols):
            descricao = raw_cols[col_indices["descricao"]]

        beneficio_raw = ""
        if col_indices["beneficio"] >= 0 and col_indices["beneficio"] < len(raw_cols):
            beneficio_raw = raw_cols[col_indices["beneficio"]]
        else:
            m = _RE_NCM_BENEFICIO.search(full_line)
            beneficio_raw = m.group(0) if m else ""

        b_lower = beneficio_raw.lower()
        if "isen" in b_lower:         beneficio = "isencao"
        elif "redu" in b_lower:       beneficio = "reducao"
        elif "difer" in b_lower:      beneficio = "diferimento"
        elif "suspen" in b_lower:     beneficio = "suspensao"
        elif "st" in b_lower or "substitui" in b_lower: beneficio = "st"
        elif "crédito" in b_lower or "credito" in b_lower: beneficio = "credito_outorgado"
        elif "não incide" in b_lower or "nao incide" in b_lower: beneficio = "nao_incidencia"
        else: beneficio = beneficio_raw or "tributado"

        percentual = ""
        if col_indices["percentual"] >= 0 and col_indices["percentual"] < len(raw_cols):
            percentual = raw_cols[col_indices["percentual"]]
        else:
            m = _RE_PERCENTUAL.search(full_line)
            percentual = m.group(0) if m else ""

        condicao = ""
        if col_indices["condicao"] >= 0 and col_indices["condicao"] < len(raw_cols):
            condicao = raw_cols[col_indices["condicao"]]

        dispositivo = ""
        if col_indices["dispositivo"] >= 0 and col_indices["dispositivo"] < len(raw_cols):
            dispositivo = raw_cols[col_indices["dispositivo"]]

        ncm_records.append({
            "ncm":         ncm_fmt,
            "ncm_norm":    ncm_norm[:8],
            "descricao":   descricao[:500] if descricao else "",
            "beneficio":   beneficio,
            "percentual":  percentual[:50] if percentual else "",
            "base_calculo": "",
            "condicao":    condicao[:500] if condicao else "",
            "dispositivo": dispositivo[:200] if dispositivo else "",
            "tipo_norma":  source_meta.get("doc_type", ""),
            "uf":          source_meta.get("uf_aplicacao", "MA"),
            "file_name":   source_meta.get("file_name", ""),
            "file_hash":   source_meta.get("file_hash", ""),
            "parent_id":   parent_id,
            "folder_name": source_meta.get("folder_name", ""),
        })

    return ncm_records


def _extract_tables(docling_result: Any, source_meta: dict) -> tuple[list[dict], list[dict]]:
    table_chunks: list[dict] = []
    ncm_records:  list[dict] = []

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
                    "content":     f"[TABELA {i + 1}]\n{tmd}",
                    "parent_id":   pid,
                    "chunk_level": "parent",
                    "chunk_index": idx,
                    "h1":          base["h1"],
                    "h2":          base["h2"],
                    "metadata":    {**base, "chunk_index": idx,
                                    "chunk_level": "parent", "parent_id": pid},
                })

                extracted = _extract_ncms_from_table(tmd, source_meta, pid)
                if extracted:
                    ncm_records.extend(extracted)
                    logger.debug(f"  📦 Tabela {i+1}: {len(extracted)} NCMs extraídos")

            except Exception as e:
                logger.debug(f"  ⚠ Tabela {i}: {e}")

    except Exception as e:
        logger.debug(f"  ⚠ Extração tabelas: {e}")

    if ncm_records:
        logger.info(f"  📦 Total NCMs extraídos das tabelas: {len(ncm_records)}")

    return table_chunks, ncm_records


def _upsert_ncm_records(ncm_records: list[dict]) -> None:
    """
    FIX #3: reutiliza cliente Supabase cacheado em vez de criar novo.
    """
    if not ncm_records:
        return
    try:
        sb = _supabase()  # FIX #3: era create_client(...) — agora usa cache
        for i in range(0, len(ncm_records), 100):
            sb.table("kb_ncm_fiscal").upsert(
                ncm_records[i: i + 100],
                on_conflict="ncm_norm,file_hash,beneficio",
            ).execute()
        logger.info(f"  ✅ {len(ncm_records)} NCMs gravados em kb_ncm_fiscal")
    except Exception as e:
        logger.error(f"  ✗ Erro ao gravar NCMs: {e}")


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


def _json_exists_in_drive(
    svc: Any,
    json_filename: str,
    folder_id: str,
) -> Optional[str]:
    """
    FIX #1: verifica se o .json já existe no Drive (não no Supabase).
    Retorna o file_id do Drive se existir, None caso contrário.
    """
    from gdrive import _get_file_id_in_folder
    return _get_file_id_in_folder(svc, json_filename, folder_id)


def _embedding_already_exists(table: str, file_hash: str) -> bool:
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
# FIX #9 — EMBEDDINGS com retry e backoff
# ══════════════════════════════════════════════════════════════════════════════

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _embed_batch_with_retry(texts: list[str]) -> list[list[float]]:
    return _embeddings().embed_documents(texts)


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

    from gdrive import _get_service, _get_or_create_folder, _upload_bytes_to_drive, _get_file_id_in_folder

    # FIX #1 — verifica re-processamento no Drive (não no Supabase)
    json_filename = file_name.rsplit(".", 1)[0] + ".json"
    svc           = _get_service()
    folder_id     = _get_or_create_folder(svc, folder_name, settings.GDRIVE_ROOT_FOLDER_ID)

    existing_json_id = _json_exists_in_drive(svc, json_filename, folder_id)
    if existing_json_id:
        # Ainda computa o hash para retornar consistente, mas não re-processa
        file_hash = _sha256(pdf_path)
        logger.info(f"  ⏭ {json_filename} já existe no Drive — pulando Fase 1")
        return {
            "status":        "json_exists",
            "file":          file_name,
            "json_file":     json_filename,
            "drive_json_id": existing_json_id,
            "file_hash":     file_hash,
        }

    file_hash = _sha256(pdf_path)

    # ── Detecta HTML disfarçado de PDF (FIX #4 dentro de _ensure_correct_extension)
    doc_path = _ensure_correct_extension(pdf_path)

    # ── Docling: documento → markdown estruturado ─────────────────────────────
    logger.info("  🔍 Docling: convertendo documento...")
    result   = _converter().convert(str(doc_path))
    markdown = result.document.export_to_markdown()
    pages    = len(result.document.pages) if result.document.pages else 0
    logger.info(f"  Docling OK: {pages} páginas | {len(markdown):,} chars")

    # ── Detecta tipo de documento (FIX #6: amostra 2.000 chars) ──────────────
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
    table_chunks, ncm_records = _extract_tables(result, source_meta)
    if table_chunks:
        parents.extend(table_chunks)
        logger.info(f"  📊 {len(table_chunks)} tabela(s) | {len(ncm_records)} NCM(s) extraídos")

    if ncm_records:
        _upsert_ncm_records(ncm_records)

    logger.info(f"  → {len(parents)} parents | {len(children)} children")

    # ── FIX #2 + #5 + #9: extração JSON paralela com log de descarte ─────────
    MAX_JSON    = 200
    LLM_WORKERS = getattr(settings, "LLM_WORKERS", 8)
    parent_jsons = _extract_json_parallel(parents, doc_type, MAX_JSON, LLM_WORKERS)

    # ── Monta payload JSON para salvar no Drive ───────────────────────────────
    chunks_payload = []
    for i, p in enumerate(parents):
        pjson      = parent_jsons[i]
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

    json_ok = sum(
        1 for j in parent_jsons[:MAX_JSON]
        if j and "_llm_error" not in j and "_parse_error" not in j
    )

    json_payload = {
        "file_name":      file_name,
        "file_id":        file_id,
        "file_hash":      file_hash,
        "folder_name":    folder_name,
        "table_name":     table_name,
        "doc_type":       doc_type.value,
        "pages":          pages,
        "total_parents":  len(parents),
        "total_children": len(children),
        "json_ok":        json_ok,
        "generated_at":   datetime.utcnow().isoformat(),
        "chunks":         chunks_payload,
    }

    # ── Salva .json no Google Drive ───────────────────────────────────────────
    json_bytes = json.dumps(json_payload, ensure_ascii=False, indent=2).encode("utf-8")

    # Remove versão anterior se existir (por segurança, embora checamos acima)
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
        "status":        "json_saved",
        "file":          file_name,
        "json_file":     json_filename,
        "drive_json_id": drive_json_id,
        "file_hash":     file_hash,
        "doc_type":      doc_type.value,
        "parents":       len(parents),
        "children":      len(children),
        "json_ok":       json_ok,
        "pages":         pages,
    }


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 — index_from_json()
# Lê .json do Drive → Embeddings incrementais → Upsert no Supabase
# FIX #10 — embeddings gravados por lote, com retomada incremental
# ══════════════════════════════════════════════════════════════════════════════

def index_from_json(
    json_filename: str,
    folder_name: str,
    table_name: str,
) -> dict[str, Any]:
    """
    Fase 2: Lê o .json do Google Drive, gera embeddings nos children
    e faz upsert incremental no Supabase.

    FIX #10: children já gravados com embedding são detectados e pulados,
    permitindo retomada em caso de falha no meio do processo.
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

    # ── Verifica se embeddings já existem (todos completos) ───────────────────
    if _embedding_already_exists(table_name, file_hash):
        logger.info(f"  ⏭ Embeddings já existem para {file_name}")
        return {"status": "already_indexed", "file": file_name}

    chunks      = payload.get("chunks", [])
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

    # ── Upsert parents primeiro (sem embedding) ───────────────────────────────
    _upsert_batch(table_name, parent_rows)
    logger.debug(f"  ✓ {len(parent_rows)} parents gravados")

    # ── FIX #10: embeddings incrementais por lote ────────────────────────────
    # Detecta quais children já têm embedding para permitir retomada
    existing_idx: set[int] = set()
    try:
        resp = (
            _supabase().table(table_name)
            .select("chunk_index")
            .eq("file_hash", file_hash)
            .eq("chunk_level", "child")
            .not_.is_("embedding", "null")
            .execute()
        )
        existing_idx = {r["chunk_index"] for r in resp.data}
        if existing_idx:
            logger.info(f"  ↩ Retomada: {len(existing_idx)} children já embedados — pulando")
    except Exception as e:
        logger.warning(f"  ⚠ Não foi possível verificar embeddings existentes: {e}")

    pending_rows  = [r for r in child_rows  if r["chunk_index"] not in existing_idx]
    pending_texts = [t for r, t in zip(child_rows, all_child_texts) if r["chunk_index"] not in existing_idx]

    if not pending_rows:
        logger.info("  ⏭ Todos os children já têm embedding")
        return {"status": "already_indexed", "file": file_name}

    logger.info(f"  🔢 Gerando {len(pending_texts)} embeddings (FIX #9: retry ativo)...")
    embedded_count = 0

    for i in range(0, len(pending_texts), settings.BATCH_SIZE):
        batch_texts = pending_texts[i: i + settings.BATCH_SIZE]
        batch_rows  = pending_rows[i: i + settings.BATCH_SIZE]

        # FIX #9: retry com backoff em cada lote de embeddings
        vecs = _embed_batch_with_retry(batch_texts)

        for row, vec in zip(batch_rows, vecs):
            row["embedding"] = vec

        # FIX #10: persiste o lote imediatamente (não acumula tudo em memória)
        _upsert_batch(table_name, batch_rows)
        embedded_count += len(batch_rows)
        logger.debug(f"    Embeddings: {embedded_count}/{len(pending_texts)} gravados")

    logger.success(f"  ✅ {embedded_count} children gravados em '{table_name}'")

    return {
        "status":   "indexed",
        "file":     file_name,
        "table":    table_name,
        "parents":  len(parent_rows),
        "children": embedded_count,
        "pages":    payload.get("pages", 0),
    }
