"""
pipeline.py — Pipeline Multi-Formato em Duas Fases v2.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMATOS SUPORTADOS (Docling processa todos):
  PDF, DOCX, PPTX, XLSX/XLS, HTML, EML, MSG, PNG, JPEG, TIFF

FASE 1 — process_document()
  Arquivo (Drive) → Docling → Chunks → LLM JSON → .json no Drive

FASE 2 — index_from_json()
  .json Drive → Embeddings → Upsert Supabase

MELHORIAS v2.0:
  ✓ Multi-formato: PDF, DOCX, XLSX, PPTX, EML/MSG, imagens
  ✓ LLM com fallback automático: OpenAI → Anthropic → Gemini
  ✓ Extrator NCM sem falsos positivos (valida prefixo capítulo)
  ✓ Detecção CEST vs NCM (Convênio 142/2018)
  ✓ DocType PLANILHA e CORRESPONDENCIA para novos formatos
  ✓ Chunking por aba (XLSX) e por thread (EML)
  ✓ Verificação se JSON já tem estruturado correto antes de skip
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
    LEGISLACAO       = "legislacao"
    CONVENIO         = "convenio"
    NORMA_TECNICA    = "norma_tecnica"
    TRABALHISTA      = "trabalhista"
    SOCIETARIO       = "societario"
    PLANILHA         = "planilha"        # XLSX/XLS/CSV
    CORRESPONDENCIA  = "correspondencia" # EML/MSG
    APRESENTACAO     = "apresentacao"    # PPTX
    IMAGEM           = "imagem"          # PNG/JPEG/TIFF
    GENERICO         = "generico"


_FOLDER_DOCTYPE: dict[str, DocType] = {
    "analista_fiscal":               DocType.LEGISLACAO,
    "analista_contabil":             DocType.NORMA_TECNICA,
    "analista_departamento_pessoal": DocType.TRABALHISTA,
    "analista_societario":           DocType.SOCIETARIO,
    "analista_abertura_empresas":    DocType.SOCIETARIO,
}

# Extensões suportadas por formato
_EXT_TO_DOCTYPE: dict[str, DocType] = {
    ".xlsx": DocType.PLANILHA,
    ".xls":  DocType.PLANILHA,
    ".csv":  DocType.PLANILHA,
    ".eml":  DocType.CORRESPONDENCIA,
    ".msg":  DocType.CORRESPONDENCIA,
    ".pptx": DocType.APRESENTACAO,
    ".ppt":  DocType.APRESENTACAO,
    ".docx": DocType.GENERICO,
    ".doc":  DocType.GENERICO,
    ".png":  DocType.IMAGEM,
    ".jpg":  DocType.IMAGEM,
    ".jpeg": DocType.IMAGEM,
    ".tiff": DocType.IMAGEM,
    ".tif":  DocType.IMAGEM,
}

_RE_CONVENIO   = re.compile(r"conv[eê]nio\s+icms|protocolo\s+icms|ajuste\s+sinief|confaz", re.I)
_RE_NORMA      = re.compile(r"\bnbc\s+t[ga]\b|\bcpc\s+\d|\bcfc\b|\bifrs\b|\bicpc\b|\bocpc\b", re.I)
_RE_ARTIGO     = re.compile(r"(?m)^\s*(?:Art(?:igo)?\.?\s*\d+[º°oa]?|§\s*\d+[º°oa]?)\s*[\.\-–—]")
_RE_CLAUSULA   = re.compile(r"(?mi)^\s*Cl[aá]usula\s+\w+")
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

    DocType.PLANILHA: """\
Você analisa planilhas fiscais/contábeis brasileiras.
Retorne SOMENTE um objeto JSON válido:
{
  "assunto": "resumo objetivo em até 12 palavras",
  "tipo_planilha": "apuracao_icms|apuracao_pis_cofins|ncm_st|folha_pagamento|dre|balanco|outro",
  "periodo_referencia": "MM/AAAA ou null",
  "empresa": "nome da empresa ou null",
  "totais": "descrição de totais relevantes ou null",
  "tributos_envolvidos": ["lista de tributos mencionados"],
  "palavras_chave": ["até 6 termos"]
}""",

    DocType.CORRESPONDENCIA: """\
Você analisa correspondências fiscais/contábeis (emails, ofícios, intimações).
Retorne SOMENTE um objeto JSON válido:
{
  "assunto": "resumo objetivo em até 12 palavras",
  "tipo": "intimacao|notificacao|consulta|resposta|oficio|auto_infracao|outro",
  "remetente": "órgão ou pessoa ou null",
  "destinatario": "órgão ou pessoa ou null",
  "data": "dd/mm/aaaa ou null",
  "orgao_emissor": "SEFAZ-MA|RFB|PGFN|JUCEMA|outro|null",
  "prazo_resposta": "prazo mencionado ou null",
  "tributo": "ICMS|IPI|PIS|COFINS|IRPJ|CSLL|ISS|todos|null",
  "numero_processo": "número do processo ou auto ou null",
  "palavras_chave": ["até 6 termos"]
}""",

    DocType.APRESENTACAO: """\
Você analisa apresentações e slides sobre temas fiscais/contábeis.
Retorne SOMENTE um objeto JSON válido:
{
  "assunto": "resumo objetivo em até 12 palavras",
  "tema_principal": "tema central da apresentação",
  "publico_alvo": "contadores|fiscais|empresarios|estudantes|todos",
  "tributos_abordados": ["lista de tributos mencionados"],
  "normas_citadas": ["leis ou normas mencionadas"],
  "palavras_chave": ["até 6 termos"]
}""",

    DocType.IMAGEM: """\
Você analisa imagens de documentos fiscais/contábeis (notas fiscais, certidões, comprovantes).
Retorne SOMENTE um objeto JSON válido:
{
  "assunto": "resumo objetivo em até 12 palavras",
  "tipo_documento": "nota_fiscal|certidao|comprovante|extrato|contrato|outro",
  "emitente": "nome do emitente ou null",
  "destinatario": "nome do destinatario ou null",
  "data": "dd/mm/aaaa ou null",
  "valor": "valor monetário principal ou null",
  "tributos": ["tributos mencionados"],
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


# ── LLM com fallback automático OpenAI → Anthropic → Gemini ─────────────────

def _make_llm_instance(provider: str):
    if provider == "openai" and settings.OPENAI_API_KEY:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", openai_api_key=settings.OPENAI_API_KEY,
                          max_tokens=600, temperature=0)
    if provider == "anthropic" and settings.ANTHROPIC_API_KEY:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-haiku-4-5-20251001",
                             anthropic_api_key=settings.ANTHROPIC_API_KEY,
                             max_tokens=600, temperature=0)
    if provider == "gemini" and settings.GEMINI_API_KEY:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                                      google_api_key=settings.GEMINI_API_KEY,
                                      max_output_tokens=600)
    return None


def _llm_providers() -> list[str]:
    order = []
    if settings.OPENAI_API_KEY:    order.append("openai")
    if settings.ANTHROPIC_API_KEY: order.append("anthropic")
    if settings.GEMINI_API_KEY:    order.append("gemini")
    if not order:
        raise RuntimeError("Nenhuma chave de LLM configurada.")
    return order


@lru_cache(maxsize=1)
def _llm():
    for p in _llm_providers():
        llm = _make_llm_instance(p)
        if llm:
            logger.debug(f"LLM pipeline: {p}")
            return llm
    raise RuntimeError("Nenhum LLM disponível.")


# ── Docling: converter multi-formato ─────────────────────────────────────────

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
    ext = Path(filename).suffix.lower()

    # Tipos determinados pela extensão (prioridade)
    if ext in _EXT_TO_DOCTYPE:
        base = _EXT_TO_DOCTYPE[ext]
        # DOCX pode ser legislação dependendo da pasta
        if ext in (".docx", ".doc"):
            base = _FOLDER_DOCTYPE.get(folder_name, DocType.GENERICO)
    else:
        base = _FOLDER_DOCTYPE.get(folder_name, DocType.GENERICO)

    probe = (filename + " " + sample[:800]).lower()
    if _RE_CONVENIO.search(probe):
        return DocType.CONVENIO
    if _RE_NORMA.search(probe):
        return DocType.NORMA_TECNICA
    return base


# ══════════════════════════════════════════════════════════════════════════════
# EXTRAÇÃO JSON VIA LLM — COM FALLBACK AUTOMÁTICO
# ══════════════════════════════════════════════════════════════════════════════

def _extract_json(content: str, doc_type: DocType) -> dict:
    from langchain_core.messages import HumanMessage, SystemMessage
    system  = _SYSTEM_PROMPTS.get(doc_type, _SYSTEM_PROMPTS[DocType.GENERICO])
    snippet = content[:3000]
    msgs = [
        SystemMessage(content=system),
        HumanMessage(content=f"TRECHO DO DOCUMENTO:\n\n{snippet}"),
    ]

    providers  = _llm_providers()
    last_error = None

    for provider in providers:
        llm = _make_llm_instance(provider)
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
            if any(k in err_str.lower() for k in [
                "credit", "balance", "quota", "rate_limit",
                "insufficient", "billing", "payment", "403", "429", "401"
            ]):
                logger.debug(f"    ↳ {provider} sem créditos — tentando próximo")
                last_error = err_str
                continue
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
    if not _is_html(path):
        return path
    html_path = path.with_suffix(".html")
    import shutil
    shutil.copy2(path, html_path)
    logger.info("  📄 HTML detectado — usando backend HTML do Docling")
    return html_path


# ══════════════════════════════════════════════════════════════════════════════
# CHUNKING POR FORMATO
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

    parents: list[dict]  = []
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


def _split_planilha(markdown: str, source_meta: dict) -> tuple[list[dict], list[dict]]:
    """Chunking por seção/aba para planilhas XLSX."""
    # Divide por separadores de aba (## nome_da_aba no markdown do Docling)
    sections = re.split(r"(?m)^#{1,3}\s+", markdown)
    parents:  list[dict] = []
    children: list[dict] = []
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHILD_SIZE, chunk_overlap=_CHILD_OVERLAP,
        separators=["\n\n", "\n", ". "],
    )
    for i, sec in enumerate(sections):
        sec = sec.strip()
        if not sec or len(sec) < 20:
            continue
        pid = str(uuid.uuid4())
        first_line = sec.split("\n")[0][:100]
        base = {**source_meta, "unit_type": "aba",
                "h1": first_line, "h2": f"Aba {i+1}"}
        parents.append({
            "content": sec, "parent_id": pid,
            "chunk_level": "parent", "chunk_index": i,
            "h1": base["h1"], "h2": base["h2"],
            "metadata": {**base, "chunk_index": i,
                         "chunk_level": "parent", "parent_id": pid},
        })
        for j, ct in enumerate(child_splitter.split_text(sec)):
            if not ct.strip():
                continue
            children.append({
                "content": ct, "parent_id": pid,
                "chunk_level": "child", "chunk_index": i * 1000 + j,
                "metadata": {**base, "chunk_index": i * 1000 + j,
                             "chunk_level": "child", "parent_id": pid},
            })
    return parents, children


def _select_strategy(
    doc_type: DocType,
    markdown: str,
    source_meta: dict,
) -> tuple[list[dict], list[dict]]:
    arts = len(_RE_ARTIGO.findall(markdown))
    clas = len(_RE_CLAUSULA.findall(markdown))
    logger.debug(f"  Artigos: {arts} | Cláusulas: {clas}")

    if doc_type == DocType.PLANILHA:
        logger.info(f"  📐 Chunking: PLANILHA")
        return _split_planilha(markdown, source_meta)

    if doc_type == DocType.CORRESPONDENCIA:
        logger.info(f"  📐 Chunking: CORRESPONDÊNCIA")
        return _split_by_sections(markdown, source_meta)

    if doc_type == DocType.APRESENTACAO:
        logger.info(f"  📐 Chunking: APRESENTAÇÃO (slides)")
        return _split_by_sections(markdown, source_meta)

    if doc_type in (DocType.LEGISLACAO, DocType.TRABALHISTA, DocType.SOCIETARIO):
        if arts > 0:
            logger.info(f"  📐 Chunking: ARTIGOS ({arts})")
            return _split_by_legal_unit(markdown, source_meta, _RE_ARTIGO, "artigo")

    if doc_type == DocType.CONVENIO:
        if clas > 0:
            logger.info(f"  📐 Chunking: CLÁUSULAS ({clas})")
            return _split_by_legal_unit(markdown, source_meta, _RE_CLAUSULA, "clausula")
        if arts > 0:
            logger.info(f"  📐 Chunking: ARTIGOS ({arts})")
            return _split_by_legal_unit(markdown, source_meta, _RE_ARTIGO, "artigo")

    if doc_type == DocType.NORMA_TECNICA:
        logger.info(f"  📐 Chunking: SEÇÕES (norma técnica)")
        return _split_by_sections(markdown, source_meta)

    if arts > 3:
        logger.info(f"  📐 Chunking: ARTIGOS genérico ({arts})")
        return _split_by_legal_unit(markdown, source_meta, _RE_ARTIGO, "artigo")

    logger.info(f"  📐 Chunking: SEÇÕES genérico")
    return _split_by_sections(markdown, source_meta)


# ══════════════════════════════════════════════════════════════════════════════
# EXTRAÇÃO NCM — SEM FALSOS POSITIVOS
# ══════════════════════════════════════════════════════════════════════════════

# NCM com separadores obrigatórios entre grupos (evita capturar CEST e datas)
_RE_NCM_STRICT = re.compile(
    r"(?<!\d)(\d{4})[.\-](\d{2})[.\-](\d{2})[.\-](\d{2})(?!\d)"  # 8 dígitos XXXX.XX.XX.XX
    r"|(?<!\d)(\d{4})[.\-](\d{2})[.\-](\d{2})(?!\d)"             # 7 dígitos XXXX.XX.XX
    r"|(?<!\d)(\d{4})[.\-](\d{2})(?!\d)"                          # 6 dígitos XXXX.XX
)
_RE_CEST = re.compile(r"\d{2}\.\d{3}\.\d{2}")  # formato CEST: XX.XXX.XX

_BENEFICIO_KEYWORDS = {
    "isencao": ["isen", "imune", "imunidade", "não incide", "nao incide",
                "não tributad", "nao tributad"],
    "reducao":  ["redu", "base reduzida", "carga reduzida"],
    "diferimento": ["difer"],
    "suspensao": ["suspen"],
    "st": ["substitui", " st ", "substituição tributária",
           "substituicao tributaria", "mva", "pauta"],
    "credito_outorgado": ["crédito outorg", "credito outorg", "crédito presumido"],
    "nao_incidencia": ["não incid", "nao incid"],
}

_HEADERS_NCM = ["ncm", "ncm/sh", "ncm_sh", "código ncm", "posição ncm", "ncm ou sh"]
_HEADERS_BENEFICIO = [
    "benefício", "beneficio", "tratamento", "situação tributária",
    "sit. tributária", "tributação", "alíquota", "aliquota",
    "modalidade", "tipo de benefício"
]


def _is_valid_ncm_prefix(ncm_norm: str) -> bool:
    """Capítulo NCM válido: 01..97."""
    if len(ncm_norm) < 2:
        return False
    try:
        return 1 <= int(ncm_norm[:2]) <= 97
    except ValueError:
        return False


def _normalize_ncm(raw: str) -> str:
    return re.sub(r"[.\-\s]", "", raw)


def _format_ncm(norm: str) -> str:
    n = norm.zfill(8)
    if len(n) >= 8:
        return f"{n[:4]}.{n[4:6]}.{n[6:8]}"
    if len(n) == 6:
        return f"{n[:4]}.{n[4:6]}"
    return norm


def _classify_beneficio(text: str) -> str:
    t = text.lower()
    for tipo, kws in _BENEFICIO_KEYWORDS.items():
        if any(k in t for k in kws):
            return tipo
    return ""


def _detect_table_context(header_cols: list[str]) -> dict:
    ctx = {
        "has_ncm": False, "has_cest": False, "has_beneficio": False,
        "is_st_table": False,
        "col_indices": {"ncm": -1, "cest": -1, "descricao": -1,
                        "beneficio": -1, "percentual": -1,
                        "condicao": -1, "dispositivo": -1},
    }
    for i, col in enumerate(header_cols):
        c = col.lower().strip()
        if any(k in c for k in _HEADERS_NCM):
            ctx["has_ncm"] = True
            ctx["col_indices"]["ncm"] = i
        elif "cest" in c:
            ctx["has_cest"] = True
            ctx["col_indices"]["cest"] = i
        elif any(k in c for k in ["descri", "produto", "mercadoria", "especificação"]):
            ctx["col_indices"]["descricao"] = i
        elif any(k in c for k in _HEADERS_BENEFICIO):
            ctx["has_beneficio"] = True
            ctx["col_indices"]["beneficio"] = i
        elif any(k in c for k in ["%", "alíquota", "aliquota", "mva", "percentual"]):
            ctx["col_indices"]["percentual"] = i
        elif any(k in c for k in ["condição", "condicao", "requisito"]):
            ctx["col_indices"]["condicao"] = i
        elif any(k in c for k in ["dispositiv", "fundamento", "base legal"]):
            ctx["col_indices"]["dispositivo"] = i

    # Convênio 142 e similares: CEST+NCM sem coluna de benefício → tudo é ST
    ctx["is_st_table"] = ctx["has_cest"] and ctx["has_ncm"] and not ctx["has_beneficio"]
    return ctx


def _extract_ncm_from_cell(cell: str) -> list[str]:
    """Extrai NCMs de uma célula ignorando CEST."""
    cell_clean = _RE_CEST.sub("", cell)
    ncms = []
    for m in _RE_NCM_STRICT.finditer(cell_clean):
        digits = "".join(g for g in m.groups() if g is not None)
        norm = digits[:8]
        if _is_valid_ncm_prefix(norm) and norm not in ncms:
            ncms.append(norm)
    return ncms


def _extract_ncms_from_table(
    table_md: str, source_meta: dict, parent_id: str
) -> list[dict]:
    """
    Extrai NCMs de tabela Markdown do Docling.
    Anti-falsos-positivos:
    - Só processa tabelas com coluna NCM identificada
    - CEST é removido antes de processar
    - Tabelas CEST+NCM → beneficio="st" (Convênio 142)
    - Benefício desconhecido → "indefinido" (nunca "tributado" falso)
    """
    ncm_records: list[dict] = []
    lines = [l for l in table_md.split("\n") if l.strip()]
    ctx = None
    header_found = False

    for line in lines:
        if "|" not in line:
            continue
        raw_cols = [c.strip() for c in line.split("|")]
        raw_cols = [c for c in raw_cols if c]  # remove vazios
        if not raw_cols:
            continue
        if all(re.match(r"^[-:\s]+$", c) for c in raw_cols if c.strip()):
            continue  # linha separadora

        if not header_found:
            ctx = _detect_table_context(raw_cols)
            header_found = True
            if not ctx["has_ncm"]:
                return []  # sem coluna NCM → ignora
            continue

        if ctx is None:
            continue

        # Extrai NCM
        ncm_idx = ctx["col_indices"]["ncm"]
        ncm_cell = raw_cols[ncm_idx] if 0 <= ncm_idx < len(raw_cols) else " ".join(raw_cols)
        ncm_norms = _extract_ncm_from_cell(ncm_cell)
        if not ncm_norms:
            continue

        # Descrição
        desc_idx = ctx["col_indices"]["descricao"]
        descricao = (raw_cols[desc_idx][:400]
                     if 0 <= desc_idx < len(raw_cols) else "")

        # Benefício
        full_line = " ".join(raw_cols)
        if ctx["is_st_table"]:
            beneficio = "st"
            percentual = ""
            dispositivo = source_meta.get("file_name", "")
        elif ctx["has_beneficio"]:
            ben_idx = ctx["col_indices"]["beneficio"]
            ben_text = raw_cols[ben_idx] if ben_idx < len(raw_cols) else ""
            beneficio = _classify_beneficio(ben_text) or "outro"
            perc_idx = ctx["col_indices"]["percentual"]
            percentual = (raw_cols[perc_idx]
                          if 0 <= perc_idx < len(raw_cols) else "")
            disp_idx = ctx["col_indices"]["dispositivo"]
            dispositivo = (raw_cols[disp_idx][:200]
                           if 0 <= disp_idx < len(raw_cols) else "")
        else:
            beneficio = _classify_beneficio(full_line) or "indefinido"
            percentual = ""
            dispositivo = ""

        perc_idx = ctx["col_indices"]["percentual"]
        if not percentual and 0 <= perc_idx < len(raw_cols):
            percentual = raw_cols[perc_idx][:50]

        cond_idx = ctx["col_indices"]["condicao"]
        condicao = (raw_cols[cond_idx][:400]
                    if 0 <= cond_idx < len(raw_cols) else "")

        for ncm_norm in ncm_norms:
            ncm_records.append({
                "ncm":          _format_ncm(ncm_norm),
                "ncm_norm":     ncm_norm[:8],
                "descricao":    descricao,
                "beneficio":    beneficio,
                "percentual":   percentual[:50] if percentual else "",
                "base_calculo": "",
                "condicao":     condicao,
                "dispositivo":  dispositivo[:200] if dispositivo else "",
                "tipo_norma":   source_meta.get("doc_type", ""),
                "uf":           source_meta.get("uf_aplicacao", "MA"),
                "file_name":    source_meta.get("file_name", ""),
                "file_hash":    source_meta.get("file_hash", ""),
                "parent_id":    parent_id,
                "folder_name":  source_meta.get("folder_name", ""),
            })

    return ncm_records


def _extract_tables(
    docling_result: Any, source_meta: dict
) -> tuple[list[dict], list[dict]]:
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
                    "content": f"[TABELA {i + 1}]\n{tmd}",
                    "parent_id": pid, "chunk_level": "parent",
                    "chunk_index": idx, "h1": base["h1"], "h2": base["h2"],
                    "metadata": {**base, "chunk_index": idx,
                                 "chunk_level": "parent", "parent_id": pid},
                })
                extracted = _extract_ncms_from_table(tmd, source_meta, pid)
                if extracted:
                    ncm_records.extend(extracted)
                    logger.debug(f"  📦 Tabela {i+1}: {len(extracted)} NCMs")
            except Exception as e:
                logger.debug(f"  ⚠ Tabela {i}: {e}")
    except Exception as e:
        logger.debug(f"  ⚠ Extração tabelas: {e}")

    if ncm_records:
        logger.info(f"  📦 Total NCMs extraídos: {len(ncm_records)}")
    return table_chunks, ncm_records


def _upsert_ncm_records(ncm_records: list[dict]) -> None:
    if not ncm_records:
        return
    try:
        # Deduplica pelo conjunto (ncm_norm, file_hash, beneficio)
        seen: set[tuple] = set()
        dedup = []
        for r in ncm_records:
            key = (r.get("ncm_norm", ""), r.get("file_hash", ""), r.get("beneficio", ""))
            if key not in seen:
                seen.add(key)
                dedup.append(r)

        from supabase import create_client
        sb = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)
        for i in range(0, len(dedup), 100):
            sb.table("kb_ncm_fiscal").upsert(
                dedup[i: i + 100],
                on_conflict="ncm_norm,file_hash,beneficio",
            ).execute()
        removed = len(ncm_records) - len(dedup)
        logger.info(f"  ✅ {len(dedup)} NCMs únicos gravados"
                    f"{f' ({removed} duplicatas removidas)' if removed else ''}")
    except Exception as e:
        logger.error(f"  ✗ Erro ao gravar NCMs: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# UTILITÁRIOS
# ══════════════════════════════════════════════════════════════════════════════

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def _json_already_exists_with_llm(table: str, file_hash: str) -> bool:
    """
    Verifica se já existe JSON com structured válido (LLM funcionou).
    Se o structured está vazio ou tem _llm_error, força reprocessamento.
    """
    resp = (
        _supabase().table(table)
        .select("metadata")
        .eq("file_hash", file_hash)
        .eq("chunk_level", "parent")
        .not_.is_("metadata", "null")
        .limit(5)
        .execute()
    )
    if not resp.data:
        return False
    # Verifica se ao menos 1 parent tem structured correto
    for row in resp.data:
        meta = row.get("metadata") or {}
        structured = meta.get("structured", {})
        if structured and "_llm_error" not in structured and "_parse_error" not in structured:
            assunto = structured.get("assunto", "")
            if assunto and not assunto.startswith("Error"):
                return True
    return False


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
# FASE 1 — process_document()
# Arquivo → Docling → Chunks → LLM JSON → .json no Drive
# ══════════════════════════════════════════════════════════════════════════════

def process_document(
    file_path: Path,
    file_name: str,
    file_id: str,
    folder_name: str,
    table_name: str,
    modified_at: str = "",
) -> dict[str, Any]:
    """
    Fase 1 multi-formato: converte qualquer arquivo suportado em chunks
    estruturados com JSON do LLM e salva .json no Google Drive.
    """
    logger.info(f"▶ [FASE 1] {file_name}")

    file_hash = _sha256(file_path)
    ext = Path(file_name).suffix.lower()

    # Verifica se JSON com LLM correto já existe
    if _json_already_exists_with_llm(table_name, file_hash):
        logger.info("  ⏭ JSON com structured válido já existe — pulando Fase 1")
        return {"status": "json_exists", "file": file_name, "file_hash": file_hash}

    # Detecta HTML disfarçado (só para PDF)
    doc_path = file_path
    if ext == ".pdf":
        doc_path = _ensure_correct_extension(file_path)

    # Docling: converte para markdown
    logger.info(f"  🔍 Docling: convertendo {ext}...")
    result   = _converter().convert(str(doc_path))
    markdown = result.document.export_to_markdown()
    pages    = len(result.document.pages) if result.document.pages else 0
    logger.info(f"  Docling OK: {pages} pág | {len(markdown):,} chars")

    # Detecta tipo de documento
    doc_type = _detect_doc_type(folder_name, file_name, markdown)
    logger.info(f"  📄 Tipo: {doc_type.value} | Formato: {ext}")

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
        "file_ext":    ext,
    }

    # Chunking inteligente por tipo
    parents, children = _select_strategy(doc_type, markdown, source_meta)

    # Tabelas Docling + extração NCM
    table_chunks, ncm_records = _extract_tables(result, source_meta)
    if table_chunks:
        parents.extend(table_chunks)
        logger.info(f"  📊 {len(table_chunks)} tabela(s) | {len(ncm_records)} NCM(s)")

    if ncm_records:
        _upsert_ncm_records(ncm_records)

    logger.info(f"  → {len(parents)} parents | {len(children)} children")

    # Extração JSON via LLM
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
            logger.debug(f"    {i+1}/{min(len(parents), MAX_JSON)} | {ok} OK")

    json_ok = sum(1 for j in parent_jsons
                  if j and "_llm_error" not in j and "_parse_error" not in j)
    logger.info(f"  ✓ JSON: {json_ok}/{len(parents)} sem erro")

    # Monta payload
    pid_to_children = {}
    for c in children:
        pid_to_children.setdefault(c["parent_id"], []).append(c)

    chunks_payload = []
    for i, p in enumerate(parents):
        pjson = parent_jsons[i]
        p_children = [
            {"chunk_index": c["chunk_index"], "content": c["content"],
             "parent_id": c["parent_id"]}
            for c in pid_to_children.get(p["parent_id"], [])
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
        "file_name":      file_name,
        "file_id":        file_id,
        "file_hash":      file_hash,
        "folder_name":    folder_name,
        "table_name":     table_name,
        "doc_type":       doc_type.value,
        "file_ext":       ext,
        "pages":          pages,
        "total_parents":  len(parents),
        "total_children": len(children),
        "json_ok":        json_ok,
        "generated_at":   datetime.utcnow().isoformat(),
        "chunks":         chunks_payload,
    }

    # Salva .json no Drive
    from gdrive import (_get_service, _get_or_create_folder,
                        _upload_bytes_to_drive, _get_file_id_in_folder)

    json_filename = Path(file_name).stem + ".json"
    json_bytes    = json.dumps(json_payload, ensure_ascii=False, indent=2).encode("utf-8")
    svc           = _get_service()
    folder_id     = _get_or_create_folder(svc, folder_name, settings.GDRIVE_ROOT_FOLDER_ID)

    old_id = _get_file_id_in_folder(svc, json_filename, folder_id)
    if old_id:
        try:
            svc.files().delete(fileId=old_id, supportsAllDrives=True).execute()
        except Exception:
            pass

    drive_json_id = _upload_bytes_to_drive(
        svc, json_bytes, json_filename, folder_id, mime_type="application/json"
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


# Alias para compatibilidade com orchestrator existente
process_pdf = process_document


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 — index_from_json()
# Lê .json do Drive → Embeddings → Upsert Supabase
# ══════════════════════════════════════════════════════════════════════════════

def index_from_json(
    json_filename: str,
    folder_name: str,
    table_name: str,
) -> dict[str, Any]:
    logger.info(f"▶ [FASE 2] {json_filename} → {table_name}")

    from gdrive import (_get_service, _get_or_create_folder,
                        download_file_bytes, _get_file_id_in_folder)

    svc       = _get_service()
    folder_id = _get_or_create_folder(svc, folder_name, settings.GDRIVE_ROOT_FOLDER_ID)

    json_file_id = _get_file_id_in_folder(svc, json_filename, folder_id)
    if not json_file_id:
        logger.error(f"  ✗ {json_filename} não encontrado no Drive")
        return {"status": "error", "file": json_filename,
                "error": "JSON não encontrado no Drive"}

    raw_bytes = download_file_bytes(svc, json_file_id)
    payload   = json.loads(raw_bytes.decode("utf-8"))

    file_hash = payload["file_hash"]
    file_name = payload["file_name"]
    doc_type  = payload.get("doc_type", "generico")

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
        "file_ext":    payload.get("file_ext", ".pdf"),
    }

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
            for field in ("assunto", "palavras_chave", "tributo",
                          "beneficio_fiscal", "tipo_planilha", "tipo"):
                if field in pjson:
                    enriched_meta[field] = pjson[field]

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

    # Embeddings em batch
    child_vecs: list[list[float]] = []
    logger.info(f"  🔢 Gerando {len(all_child_texts)} embeddings...")
    for i in range(0, len(all_child_texts), settings.BATCH_SIZE):
        batch = all_child_texts[i: i + settings.BATCH_SIZE]
        child_vecs.extend(_embeddings().embed_documents(batch))
        logger.debug(
            f"    Embeddings: "
            f"{min(i + settings.BATCH_SIZE, len(all_child_texts))}/{len(all_child_texts)}"
        )

    for row, vec in zip(child_rows, child_vecs):
        row["embedding"] = vec

    _upsert_batch(table_name, parent_rows)
    logger.debug(f"  ✓ {len(parent_rows)} parents gravados")

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
