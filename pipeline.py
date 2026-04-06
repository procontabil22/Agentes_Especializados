"""
pipeline.py — Pipeline Avançado
PDF → Docling (estrutura rica) → Chunking Inteligente por Tipo de Documento
→ Extração JSON Estruturada via LLM → Embeddings → Supabase (pgvector)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ESTRATÉGIAS DE CHUNKING (adaptativas por tipo de documento):

  LEGISLAÇÃO  (leis, decretos, RICMS, CTN, CLT)
    → Divide por ARTIGO (Art. Xº)
    → Parent  = texto completo do artigo (incisos, parágrafos incluídos)
    → Children = cada parágrafo/inciso individualmente

  CONVÊNIO    (CONFAZ, protocolos, ajustes SINIEF)
    → Divide por CLÁUSULA
    → Parent  = texto completo da cláusula
    → Children = subdivisões da cláusula

  NORMA TÉCNICA (NBC TG, CPC, IFRS, ITG)
    → Divide por CABEÇALHOS Markdown (H1/H2/H3)
    → Parent  = seção completa do pronunciamento
    → Children = parágrafos da seção

  TRABALHISTA (CLT, NRs, previdência)
    → Divide por ARTIGO (mesma lógica da legislação)

  SOCIETÁRIO  (Código Civil, Lei S.A., DREI)
    → Divide por ARTIGO

  GENÉRICO    (fallback)
    → Seções por cabeçalhos → parent-child por tamanho

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXTRAÇÃO JSON ESTRUTURADA (LLM por tipo):

  LEGISLAÇÃO   → tipo_norma, artigo, inciso, beneficio_fiscal, tributo, uf
  CONVÊNIO     → numero_convenio, clausula, tipo, estados, produto, aliquota
  NORMA TÉC.   → norma, tipo_orientacao, metodo, paragrafo
  TRABALHISTA  → tipo_norma, artigo, direito, beneficiario, prazo
  SOCIETÁRIO   → tipo_norma, artigo, tipo_societario, fase, orgao_registro

  JSON salvo em metadata["structured"] de cada parent.
  Children herdam o JSON do parent via parent_id.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TABELAS DOCLING:
  Tabelas extraídas como parents especiais (unit_type="tabela").
  Preservam estrutura original para consultas sobre alíquotas, MVA, etc.
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


# Mapeamento pasta Drive → tipo base
_FOLDER_DOCTYPE: dict[str, DocType] = {
    "analista_fiscal":               DocType.LEGISLACAO,
    "analista_contabil":             DocType.NORMA_TECNICA,
    "analista_departamento_pessoal": DocType.TRABALHISTA,
    "analista_societario":           DocType.SOCIETARIO,
    "analista_abertura_empresas":    DocType.SOCIETARIO,
}

# Padrões de detecção por conteúdo
_RE_CONVENIO  = re.compile(r"conv[eê]nio\s+icms|protocolo\s+icms|ajuste\s+sinief|confaz", re.I)
_RE_NORMA     = re.compile(r"\bnbc\s+t[ga]\b|\bcpc\s+\d|\bcfc\b|\bifrs\b|\bicpc\b|\bocpc\b", re.I)
_RE_ARTIGO    = re.compile(
    r"(?m)^\s*(?:Art(?:igo)?\.?\s*\d+[º°oa]?|§\s*\d+[º°oa]?)\s*[.\-–—]",
)
_RE_CLAUSULA  = re.compile(r"(?mi)^\s*Cl[aá]usula\s+\w+")
_RE_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


# ══════════════════════════════════════════════════════════════════════════════
# PROMPTS DE EXTRAÇÃO JSON POR TIPO
# ══════════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPTS: dict[DocType, str] = {

    DocType.LEGISLACAO: """\
Você é especialista em legislação tributária brasileira (ICMS, IPI, PIS, COFINS, IRPJ, CSLL, ISS).
Analise o trecho e retorne SOMENTE um objeto JSON válido, sem markdown nem explicações:
{
  "tipo_norma": "lei|decreto|regulamento|instrucao_normativa|portaria|resolucao|emenda",
  "numero_norma": "número/ano ex: 7.799/2002 ou null",
  "artigo": "ex: Art. 4º ou null",
  "inciso": "ex: II ou null",
  "paragrafo": "ex: § 3º ou null",
  "assunto": "resumo objetivo em até 12 palavras",
  "beneficio_fiscal": {
    "tipo": "isencao|reducao_bc|diferimento|credito_outorgado|st|imunidade|nenhum",
    "produto_operacao": "produto/serviço/operação beneficiada ou null",
    "percentual": "ex: 100% ou 41.18% ou null",
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
Analise o trecho e retorne SOMENTE um objeto JSON válido:
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
Você é especialista em normas contábeis brasileiras (NBC TG, CPC, ITG, IFRS, ICPC, OCPC).
Analise o trecho e retorne SOMENTE um objeto JSON válido:
{
  "norma": "ex: NBC TG 26 / CPC 26 ou null",
  "item_paragrafo": "número do item ou parágrafo ex: item 15 ou null",
  "assunto": "resumo objetivo em até 12 palavras",
  "tipo_orientacao": "objetivo|alcance|reconhecimento|mensuracao|divulgacao|apresentacao|definicao|transicao",
  "aplica_se_a": "PME|grande|entidade_sem_fins|todas ou descrição ou null",
  "metodo_criterio": "método ou critério contábil principal ou null",
  "conta_elemento": "nome da conta ou elemento patrimonial ou null",
  "vigencia": "data de vigência ou null",
  "palavras_chave": ["até 6 termos"]
}""",

    DocType.TRABALHISTA: """\
Você é especialista em direito do trabalho, previdência social e eSocial.
Analise o trecho e retorne SOMENTE um objeto JSON válido:
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
Você é especialista em direito empresarial, registros mercantis e abertura de empresas no Brasil.
Analise o trecho e retorne SOMENTE um objeto JSON válido:
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
Analise o trecho do documento abaixo e retorne SOMENTE um objeto JSON válido:
{
  "assunto": "resumo objetivo em até 12 palavras",
  "tipo_conteudo": "definicao|regra|procedimento|tabela|exemplo|outro",
  "normas_citadas": ["leis, decretos ou normas mencionadas — lista vazia se nenhuma"],
  "entidades": ["organizações ou órgãos mencionados — lista vazia se nenhum"],
  "palavras_chave": ["até 6 termos"]
}""",
}


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO DE CHUNKS
# ══════════════════════════════════════════════════════════════════════════════

_PARENT_SIZE    = settings.CHUNK_SIZE * 12   # ~12 000 chars — artigo completo
_PARENT_OVERLAP = 100
_CHILD_SIZE     = settings.CHUNK_SIZE * 2    # ~2 000 chars  — parágrafo/inciso
_CHILD_OVERLAP  = settings.CHUNK_OVERLAP * 2
_ARTICLE_MAX    = settings.CHUNK_SIZE * 8    # artigo muito longo → sub-parents


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
    """LLM para extração de metadados JSON — temperatura 0 para consistência."""
    if settings.ANTHROPIC_API_KEY:
        from langchain_anthropic import ChatAnthropic
        logger.debug("LLM extração: Claude Haiku")
        return ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            anthropic_api_key=settings.ANTHROPIC_API_KEY,
            max_tokens=600,
            temperature=0,
        )
    if settings.OPENAI_API_KEY:
        from langchain_openai import ChatOpenAI
        logger.debug("LLM extração: GPT-4o-mini")
        return ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=settings.OPENAI_API_KEY,
            max_tokens=600,
            temperature=0,
        )
    if settings.GEMINI_API_KEY:
        from langchain_google_genai import ChatGoogleGenerativeAI
        logger.debug("LLM extração: Gemini Flash")
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
    opts.do_ocr = True                           # OCR em documentos escaneados
    opts.do_table_structure = True               # Extração estrutural de tabelas
    opts.table_structure_options.do_cell_matching = True
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )


# ══════════════════════════════════════════════════════════════════════════════
# DETECÇÃO DE TIPO DE DOCUMENTO
# ══════════════════════════════════════════════════════════════════════════════

def _detect_doc_type(folder_name: str, filename: str, sample: str) -> DocType:
    """
    Detecta o tipo de documento combinando:
      1. folder_name (sinal mais forte)
      2. filename (sinal médio)
      3. amostra do conteúdo (sinal fino)
    """
    base = _FOLDER_DOCTYPE.get(folder_name, DocType.GENERICO)
    probe = (filename + " " + sample[:800]).lower()

    # Convênio/protocolo supera qualquer pasta
    if _RE_CONVENIO.search(probe):
        return DocType.CONVENIO

    # Norma técnica contábil
    if _RE_NORMA.search(probe):
        return DocType.NORMA_TECNICA

    return base


# ══════════════════════════════════════════════════════════════════════════════
# EXTRAÇÃO JSON VIA LLM
# ══════════════════════════════════════════════════════════════════════════════

def _extract_json(content: str, doc_type: DocType) -> dict:
    """
    Envia até 3 000 chars do chunk para o LLM e retorna metadados JSON.
    Em caso de erro retorna dict mínimo sem lançar exceção.
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    system = _SYSTEM_PROMPTS.get(doc_type, _SYSTEM_PROMPTS[DocType.GENERICO])
    snippet = content[:3000]
    try:
        resp = _llm().invoke([
            SystemMessage(content=system),
            HumanMessage(content=f"TRECHO DO DOCUMENTO:\n\n{snippet}"),
        ])
        raw = resp.content.strip()
        # Remove fences markdown se o LLM as incluir
        fence = _RE_JSON_FENCE.search(raw)
        if fence:
            raw = fence.group(1)
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.debug(f"    ⚠ JSON inválido do LLM: {e}")
        return {"assunto": content[:80], "_parse_error": str(e)}
    except Exception as e:
        logger.debug(f"    ⚠ LLM extraction falhou: {e}")
        return {"assunto": content[:80], "_llm_error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# CHUNKING POR UNIDADES LEGAIS (Artigos / Cláusulas)
# ══════════════════════════════════════════════════════════════════════════════

def _split_by_legal_unit(
    markdown: str,
    source_meta: dict,
    pattern: re.Pattern,
    unit_name: str,
) -> tuple[list[dict], list[dict]]:
    """
    Divide o markdown nas posições de cada unidade legal detectada pelo pattern.

    Parent  = texto completo da unidade (artigo, cláusula)
    Children = subdivisões do texto (parágrafos, incisos, alíneas)
    """
    boundaries = [m.start() for m in pattern.finditer(markdown)]
    if not boundaries:
        return _split_by_sections(markdown, source_meta)   # fallback

    boundaries.append(len(markdown))

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHILD_SIZE,
        chunk_overlap=_CHILD_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    sub_splitter = RecursiveCharacterTextSplitter(
        chunk_size=_PARENT_SIZE,
        chunk_overlap=_PARENT_OVERLAP,
        separators=["\n\n", "\n", ". "],
    )

    parents:  list[dict] = []
    children: list[dict] = []
    parent_idx = child_idx = 0

    for i in range(len(boundaries) - 1):
        unit_text = markdown[boundaries[i]: boundaries[i + 1]].strip()
        if not unit_text:
            continue

        first_line = unit_text.split("\n")[0].strip()
        art_m = re.match(r"(?:Art\.?\s*(\d+[º°oa]?)|Cl[aá]usula\s+(\w+))", first_line, re.I)
        unit_num = (art_m.group(1) or art_m.group(2)) if art_m else str(i + 1)

        parent_base = {
            **source_meta,
            "unit_type":   unit_name,
            "unit_number": unit_num,
            "unit_title":  first_line[:200],
        }

        def _make_parent(text: str) -> str:
            pid = str(uuid.uuid4())
            nonlocal parent_idx
            parents.append({
                "content":     text,
                "parent_id":   pid,
                "chunk_level": "parent",
                "chunk_index": parent_idx,
                "metadata":    {**parent_base, "chunk_index": parent_idx,
                                "chunk_level": "parent", "parent_id": pid},
            })
            parent_idx += 1
            return pid

        def _make_children(text: str, pid: str) -> None:
            nonlocal child_idx
            for child_text in child_splitter.split_text(text):
                if not child_text.strip():
                    continue
                children.append({
                    "content":     child_text,
                    "parent_id":   pid,
                    "chunk_level": "child",
                    "chunk_index": child_idx,
                    "metadata":    {**parent_base, "chunk_index": child_idx,
                                    "chunk_level": "child", "parent_id": pid},
                })
                child_idx += 1

        if len(unit_text) > _ARTICLE_MAX:
            # Artigo muito extenso: divide em sub-parents
            for sub_text in sub_splitter.split_text(unit_text):
                pid = _make_parent(sub_text)
                _make_children(sub_text, pid)
        else:
            pid = _make_parent(unit_text)
            _make_children(unit_text, pid)

    return parents, children


# ══════════════════════════════════════════════════════════════════════════════
# CHUNKING POR SEÇÕES / CABEÇALHOS (Normas Técnicas e Genérico)
# ══════════════════════════════════════════════════════════════════════════════

def _split_by_sections(
    markdown: str,
    source_meta: dict,
) -> tuple[list[dict], list[dict]]:
    """
    Divide pelo cabeçalho Markdown (H1/H2/H3) usando LangChain.
    Ideal para normas técnicas (NBC, CPC) e documentos sem artigos numerados.
    """
    h_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )
    sections = h_splitter.split_text(markdown)

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=_PARENT_SIZE,
        chunk_overlap=_PARENT_OVERLAP,
        separators=["\n\n\n", "\n\n", "\n", ". "],
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHILD_SIZE,
        chunk_overlap=_CHILD_OVERLAP,
        separators=["\n\n", "\n", ". "],
    )

    parents:  list[dict] = []
    children: list[dict] = []
    parent_idx = child_idx = 0

    for parent_doc in parent_splitter.split_documents(sections):
        pid = str(uuid.uuid4())
        parent_base = {
            **source_meta,
            "unit_type": "secao",
            "h1":        parent_doc.metadata.get("h1", ""),
            "h2":        parent_doc.metadata.get("h2", ""),
            "h3":        parent_doc.metadata.get("h3", ""),
        }
        parents.append({
            "content":     parent_doc.page_content,
            "parent_id":   pid,
            "chunk_level": "parent",
            "chunk_index": parent_idx,
            "metadata":    {**parent_base, "chunk_index": parent_idx,
                            "chunk_level": "parent", "parent_id": pid},
        })
        parent_idx += 1

        for child_text in child_splitter.split_text(parent_doc.page_content):
            if not child_text.strip():
                continue
            children.append({
                "content":     child_text,
                "parent_id":   pid,
                "chunk_level": "child",
                "chunk_index": child_idx,
                "metadata":    {**parent_base, "chunk_index": child_idx,
                                "chunk_level": "child", "parent_id": pid},
            })
            child_idx += 1

    return parents, children


# ══════════════════════════════════════════════════════════════════════════════
# EXTRAÇÃO DE TABELAS DO DOCLING
# ══════════════════════════════════════════════════════════════════════════════

def _extract_tables(docling_result: Any, source_meta: dict) -> list[dict]:
    """
    Extrai tabelas do documento Docling como parents especiais.
    Essencial para documentos fiscais: tabelas de alíquotas, MVA, pauta fiscal.
    """
    table_chunks: list[dict] = []
    try:
        for i, table in enumerate(docling_result.document.tables or []):
            try:
                table_md = (
                    table.export_to_markdown()
                    if hasattr(table, "export_to_markdown")
                    else str(table)
                )
                if not table_md.strip() or len(table_md) < 20:
                    continue

                pid = str(uuid.uuid4())
                idx = 90000 + i
                table_meta = {
                    **source_meta,
                    "unit_type":   "tabela",
                    "table_index": i,
                    "is_table":    True,
                    "h1":          source_meta.get("file_name", ""),
                    "h2":          f"Tabela {i + 1}",
                }
                table_chunks.append({
                    "content":     f"[TABELA {i + 1}]\n{table_md}",
                    "parent_id":   pid,
                    "chunk_level": "parent",
                    "chunk_index": idx,
                    "metadata":    {**table_meta, "chunk_index": idx,
                                    "chunk_level": "parent", "parent_id": pid},
                })
            except Exception as e:
                logger.debug(f"  ⚠ Tabela {i} ignorada: {e}")
    except Exception as e:
        logger.debug(f"  ⚠ Extração de tabelas: {e}")
    return table_chunks


# ══════════════════════════════════════════════════════════════════════════════
# SELEÇÃO AUTOMÁTICA DA ESTRATÉGIA DE CHUNKING
# ══════════════════════════════════════════════════════════════════════════════

def _select_strategy(
    doc_type: DocType,
    markdown: str,
    source_meta: dict,
) -> tuple[list[dict], list[dict]]:
    """
    Escolhe a melhor estratégia com base no tipo detectado e no conteúdo real.
    Conta artigos e cláusulas para confirmar a escolha.
    """
    n_artigos   = len(_RE_ARTIGO.findall(markdown))
    n_clausulas = len(_RE_CLAUSULA.findall(markdown))

    logger.debug(f"  Artigos detectados: {n_artigos} | Cláusulas: {n_clausulas}")

    if doc_type == DocType.CONVENIO or (n_clausulas > 3 and n_clausulas >= n_artigos):
        logger.info(f"  📐 Chunking: CLÁUSULAS ({n_clausulas})")
        return _split_by_legal_unit(markdown, source_meta, _RE_CLAUSULA, "clausula")

    if doc_type in (DocType.LEGISLACAO, DocType.TRABALHISTA, DocType.SOCIETARIO) and n_artigos > 3:
        logger.info(f"  📐 Chunking: ARTIGOS ({n_artigos})")
        return _split_by_legal_unit(markdown, source_meta, _RE_ARTIGO, "artigo")

    if doc_type == DocType.NORMA_TECNICA:
        logger.info("  📐 Chunking: SEÇÕES (norma técnica)")
        return _split_by_sections(markdown, source_meta)

    # Fallback inteligente: tenta artigos antes de seções
    if n_artigos > 3:
        logger.info(f"  📐 Chunking: ARTIGOS fallback ({n_artigos})")
        return _split_by_legal_unit(markdown, source_meta, _RE_ARTIGO, "artigo")

    logger.info("  📐 Chunking: SEÇÕES genérico")
    return _split_by_sections(markdown, source_meta)


# ══════════════════════════════════════════════════════════════════════════════
# UTILITÁRIOS DE PERSISTÊNCIA
# ══════════════════════════════════════════════════════════════════════════════

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def _already_indexed(table: str, file_hash: str) -> bool:
    resp = (
        _supabase().table(table)
        .select("id")
        .eq("file_hash", file_hash)
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


def _build_row(
    chunk: dict,
    embedding: Optional[list[float]],
    source_meta: dict,
    structured_json: dict,
) -> dict:
    """
    Monta a linha para o Supabase.
    structured_json é salvo dentro de metadata["structured"].
    """
    meta = chunk["metadata"]
    enriched = {**meta}
    if structured_json:
        enriched["structured"] = structured_json
        # Promove campos-chave para o topo do metadata para facilitar filtros
        if "assunto" in structured_json:
            enriched["assunto"] = structured_json["assunto"]
        if "palavras_chave" in structured_json:
            enriched["palavras_chave"] = structured_json["palavras_chave"]
        if "tributo" in structured_json:
            enriched["tributo"] = structured_json["tributo"]
        if "beneficio_fiscal" in structured_json:
            enriched["beneficio_fiscal"] = structured_json["beneficio_fiscal"]

    row = {
        "content":     chunk["content"],
        "metadata":    enriched,
        "file_name":   source_meta["file_name"],
        "file_hash":   source_meta["file_hash"],
        "folder":      source_meta["folder_name"],
        "agent":       source_meta["agent"],
        "chunk_index": chunk["chunk_index"],
        "chunk_level": chunk["chunk_level"],
        "parent_id":   chunk["parent_id"],
        "h1":          meta.get("h1") or meta.get("unit_title", "")[:100],
        "h2":          meta.get("h2") or meta.get("unit_type", ""),
        "indexed_at":  source_meta["indexed_at"],
    }
    if embedding is not None:
        row["embedding"] = embedding
    return row


# ══════════════════════════════════════════════════════════════════════════════
# PONTO DE ENTRADA PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def process_pdf(
    pdf_path: Path,
    file_name: str,
    file_id: str,
    folder_name: str,
    table_name: str,
    modified_at: str = "",
) -> dict[str, Any]:
    logger.info(f"▶ {file_name} → {table_name}")

    # ── 1. Deduplicação por hash SHA-256 ──────────────────────────────────────
    file_hash = _sha256(pdf_path)
    if _already_indexed(table_name, file_hash):
        logger.info("  ⏭ Já indexado (mesmo hash)")
        return {"status": "skipped", "file": file_name}

    # ── 2. Docling: PDF → documento estruturado rico ──────────────────────────
    logger.info("  🔍 Docling: convertendo PDF...")
    result   = _converter().convert(str(pdf_path))
    markdown = result.document.export_to_markdown()
    pages    = len(result.document.pages) if result.document.pages else 0
    logger.info(f"  Docling OK: {pages} páginas | {len(markdown):,} chars markdown")

    # ── 3. Detecta tipo de documento ──────────────────────────────────────────
    doc_type = _detect_doc_type(folder_name, file_name, markdown)
    logger.info(f"  📄 Tipo detectado: {doc_type.value}")

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

    # ── 4. Chunking inteligente por tipo ──────────────────────────────────────
    parents, children = _select_strategy(doc_type, markdown, source_meta)

    # ── 5. Tabelas Docling como parents especiais ─────────────────────────────
    table_chunks = _extract_tables(result, source_meta)
    if table_chunks:
        parents.extend(table_chunks)
        logger.info(f"  📊 {len(table_chunks)} tabela(s) extraída(s) pelo Docling")

    logger.info(f"  → {len(parents)} parents | {len(children)} children")

    # ── 6. Extração JSON via LLM para cada parent ─────────────────────────────
    # Limita a 200 parents para documentos muito longos (economia de tokens)
    MAX_JSON_PARENTS = 200
    logger.info(f"  🤖 Extraindo JSON estruturado ({min(len(parents), MAX_JSON_PARENTS)} parents)...")

    parent_jsons: list[dict] = []
    for i, p in enumerate(parents):
        if i >= MAX_JSON_PARENTS:
            # Parents além do limite recebem JSON vazio
            parent_jsons.append({})
            continue
        jmeta = _extract_json(p["content"], doc_type)
        parent_jsons.append(jmeta)
        if (i + 1) % 20 == 0:
            ok = sum(1 for j in parent_jsons if j and "_llm_error" not in j and "_parse_error" not in j)
            logger.debug(f"    JSON: {i + 1}/{min(len(parents), MAX_JSON_PARENTS)} | {ok} OK")

    json_ok = sum(1 for j in parent_jsons if j and "_llm_error" not in j and "_parse_error" not in j)
    logger.info(f"  ✓ JSON extraído: {json_ok}/{len(parents)} parents sem erro")

    # ── 7. Embeddings nos children (via LangChain + OpenAI) ───────────────────
    child_texts = [c["content"] for c in children]
    child_vecs: list[list[float]] = []
    logger.info(f"  🔢 Gerando {len(child_texts)} embeddings...")
    for i in range(0, len(child_texts), settings.BATCH_SIZE):
        batch = child_texts[i: i + settings.BATCH_SIZE]
        child_vecs.extend(_embeddings().embed_documents(batch))
        logger.debug(f"    Embeddings: {min(i + settings.BATCH_SIZE, len(child_texts))}/{len(child_texts)}")

    # ── 8. Mapa parent_id → JSON para herança pelos children ──────────────────
    pid_to_json: dict[str, dict] = {
        p["parent_id"]: parent_jsons[i]
        for i, p in enumerate(parents)
    }

    # ── 9. Upsert parents (sem embedding, com JSON estruturado) ───────────────
    parent_rows = [
        _build_row(p, None, source_meta, parent_jsons[i])
        for i, p in enumerate(parents)
    ]
    _upsert_batch(table_name, parent_rows)
    logger.debug(f"  ✓ {len(parent_rows)} parents gravados")

    # ── 10. Upsert children (com embedding + JSON herdado do parent) ──────────
    child_rows = [
        _build_row(c, v, source_meta, pid_to_json.get(c["parent_id"], {}))
        for c, v in zip(children, child_vecs)
    ]
    _upsert_batch(table_name, child_rows)
    logger.success(f"  ✓ {len(child_rows)} children gravados em '{table_name}'")

    return {
        "status":          "ok",
        "file":            file_name,
        "table":           table_name,
        "doc_type":        doc_type.value,
        "parents":         len(parent_rows),
        "children":        len(child_rows),
        "tables":          len(table_chunks),
        "pages":           pages,
        "json_extracted":  json_ok,
    }
