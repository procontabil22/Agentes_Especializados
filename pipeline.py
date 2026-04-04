"""
app/pipeline.py — Pipeline: PDF → Docling → chunks → embeddings → Supabase
Clientes inicializados lazy (só quando chamados pela primeira vez).
"""
import hashlib
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from loguru import logger

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


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _already_indexed(table: str, file_hash: str) -> bool:
    resp = (
        _supabase().table(table)
        .select("id").eq("file_hash", file_hash).limit(1).execute()
    )
    return len(resp.data) > 0


def _split_markdown(markdown: str, meta: dict) -> list[dict]:
    h_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )
    c_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE * 4,
        chunk_overlap=settings.CHUNK_OVERLAP * 4,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for i, doc in enumerate(c_splitter.split_documents(h_splitter.split_text(markdown))):
        chunks.append({
            "content": doc.page_content,
            "metadata": {
                **meta, "chunk_index": i,
                "h1": doc.metadata.get("h1", ""),
                "h2": doc.metadata.get("h2", ""),
                "h3": doc.metadata.get("h3", ""),
            }
        })
    return chunks


def process_pdf(
    pdf_path: Path, file_name: str, file_id: str,
    folder_name: str, table_name: str, modified_at: str = "",
) -> dict[str, Any]:
    logger.info(f"▶ {file_name} → {table_name}")

    file_hash = _sha256(pdf_path)
    if _already_indexed(table_name, file_hash):
        logger.info("  ⏭ Já indexado")
        return {"status": "skipped", "file": file_name}

    # 1. Docling
    result = _converter().convert(str(pdf_path))
    markdown = result.document.export_to_markdown()
    pages = len(result.document.pages) if result.document.pages else 0

    # 2. Chunks
    source_meta = {
        "file_name": file_name, "file_id": file_id, "file_hash": file_hash,
        "folder_name": folder_name, "page_count": pages,
        "modified_at": modified_at,
        "indexed_at": datetime.utcnow().isoformat(),
        "agent": folder_name,
    }
    chunks = _split_markdown(markdown, source_meta)
    logger.info(f"  → {len(chunks)} chunks, {pages} páginas")

    # 3. Embeddings
    texts = [c["content"] for c in chunks]
    vecs = []
    for i in range(0, len(texts), settings.BATCH_SIZE):
        vecs.extend(_embeddings().embed_documents(texts[i: i + settings.BATCH_SIZE]))

    # 4. Upsert Supabase
    rows = [{
        "content": c["content"], "embedding": v,
        "metadata": c["metadata"],
        "file_name": c["metadata"]["file_name"],
        "file_hash": c["metadata"]["file_hash"],
        "folder": c["metadata"]["folder_name"],
        "agent": c["metadata"]["agent"],
        "chunk_index": c["metadata"]["chunk_index"],
        "h1": c["metadata"].get("h1", ""),
        "h2": c["metadata"].get("h2", ""),
        "indexed_at": c["metadata"]["indexed_at"],
    } for c, v in zip(chunks, vecs)]

    for i in range(0, len(rows), 100):
        _supabase().table(table_name).upsert(
            rows[i: i + 100], on_conflict="file_hash,chunk_index"
        ).execute()

    logger.success(f"  ✓ {len(chunks)} chunks gravados em '{table_name}'")
    return {"status": "ok", "file": file_name, "table": table_name,
            "chunks": len(chunks), "pages": pages}
