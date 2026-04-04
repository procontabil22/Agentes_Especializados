"""
pipeline.py — Pipeline: PDF → Docling → chunks parent-child → embeddings → Supabase

Estratégia Parent-Child:
  • PARENT  (~3 000 chars) — bloco semântico completo (artigo, seção, cláusula).
            Armazenado SEM embedding. Enviado ao LLM como contexto rico.
  • CHILD   (~500  chars) — subdivisão do parent, COM embedding.
            Usado na busca vetorial por similaridade.
            Aponta para o parent via parent_id.
"""

import hashlib
import uuid
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from loguru import logger

from settings import settings  # ← flat import

# ── Tamanhos dos chunks ───────────────────────────────────────────────────────
_PARENT_SIZE    = settings.CHUNK_SIZE * 12
_PARENT_OVERLAP = 100
_CHILD_SIZE     = settings.CHUNK_SIZE * 2
_CHILD_OVERLAP  = settings.CHUNK_OVERLAP * 2


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


# ── Utilitários ───────────────────────────────────────────────────────────────

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
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


# ── Chunking parent-child ─────────────────────────────────────────────────────

def _split_markdown(markdown: str, source_meta: dict) -> tuple[list[dict], list[dict]]:
    h_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )
    sections = h_splitter.split_text(markdown)

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=_PARENT_SIZE,
        chunk_overlap=_PARENT_OVERLAP,
        separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""],
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHILD_SIZE,
        chunk_overlap=_CHILD_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    parents: list[dict] = []
    children: list[dict] = []
    parent_idx = 0
    child_idx  = 0

    for parent_doc in parent_splitter.split_documents(sections):
        pid = str(uuid.uuid4())

        parent_base_meta = {
            **source_meta,
            "h1": parent_doc.metadata.get("h1", ""),
            "h2": parent_doc.metadata.get("h2", ""),
            "h3": parent_doc.metadata.get("h3", ""),
        }

        parents.append({
            "content":     parent_doc.page_content,
            "parent_id":   pid,
            "chunk_level": "parent",
            "chunk_index": parent_idx,
            "metadata": {
                **parent_base_meta,
                "chunk_index": parent_idx,
                "chunk_level": "parent",
                "parent_id":   pid,
            },
        })
        parent_idx += 1

        for text in child_splitter.split_text(parent_doc.page_content):
            if not text.strip():
                continue
            children.append({
                "content":     text,
                "parent_id":   pid,
                "chunk_level": "child",
                "chunk_index": child_idx,
                "metadata": {
                    **parent_base_meta,
                    "chunk_index": child_idx,
                    "chunk_level": "child",
                    "parent_id":   pid,
                },
            })
            child_idx += 1

    return parents, children


# ── Upsert helpers ────────────────────────────────────────────────────────────

def _upsert_batch(table: str, rows: list[dict]) -> None:
    for i in range(0, len(rows), 100):
        _supabase().table(table).upsert(
            rows[i: i + 100],
            on_conflict="file_hash,chunk_index,chunk_level",
        ).execute()


def _build_row(chunk: dict, embedding: list[float] | None, source_meta: dict) -> dict:
    meta = chunk["metadata"]
    row = {
        "content":     chunk["content"],
        "metadata":    meta,
        "file_name":   source_meta["file_name"],
        "file_hash":   source_meta["file_hash"],
        "folder":      source_meta["folder_name"],
        "agent":       source_meta["agent"],
        "chunk_index": chunk["chunk_index"],
        "chunk_level": chunk["chunk_level"],
        "parent_id":   chunk["parent_id"],
        "h1":          meta.get("h1", ""),
        "h2":          meta.get("h2", ""),
        "indexed_at":  source_meta["indexed_at"],
    }
    if embedding is not None:
        row["embedding"] = embedding
    return row


# ── Ponto de entrada principal ────────────────────────────────────────────────

def process_pdf(
    pdf_path: Path,
    file_name: str,
    file_id: str,
    folder_name: str,
    table_name: str,
    modified_at: str = "",
) -> dict[str, Any]:
    logger.info(f"▶ {file_name} → {table_name}")

    file_hash = _sha256(pdf_path)
    if _already_indexed(table_name, file_hash):
        logger.info("  ⏭ Já indexado (mesmo hash)")
        return {"status": "skipped", "file": file_name}

    # 1. Docling: PDF → Markdown estruturado
    result   = _converter().convert(str(pdf_path))
    markdown = result.document.export_to_markdown()
    pages    = len(result.document.pages) if result.document.pages else 0

    # 2. Chunking parent-child
    source_meta = {
        "file_name":   file_name,
        "file_id":     file_id,
        "file_hash":   file_hash,
        "folder_name": folder_name,
        "page_count":  pages,
        "modified_at": modified_at,
        "indexed_at":  datetime.utcnow().isoformat(),
        "agent":       folder_name,
    }

    parents, children = _split_markdown(markdown, source_meta)
    logger.info(f"  → {len(parents)} parents | {len(children)} children | {pages} páginas")

    # 3. Embeddings — somente nos children
    child_texts = [c["content"] for c in children]
    child_vecs: list[list[float]] = []

    for i in range(0, len(child_texts), settings.BATCH_SIZE):
        batch = child_texts[i: i + settings.BATCH_SIZE]
        child_vecs.extend(_embeddings().embed_documents(batch))
        logger.debug(f"  Embeddings: {min(i + settings.BATCH_SIZE, len(child_texts))}/{len(child_texts)}")

    # 4. Upsert parents (sem embedding)
    parent_rows = [_build_row(p, None, source_meta) for p in parents]
    _upsert_batch(table_name, parent_rows)
    logger.debug(f"  ✓ {len(parent_rows)} parents gravados")

    # 5. Upsert children (com embedding)
    child_rows = [_build_row(c, v, source_meta) for c, v in zip(children, child_vecs)]
    _upsert_batch(table_name, child_rows)
    logger.success(f"  ✓ {len(child_rows)} children gravados em '{table_name}'")

    return {
        "status":   "ok",
        "file":     file_name,
        "table":    table_name,
        "parents":  len(parents),
        "children": len(children),
        "pages":    pages,
    }
