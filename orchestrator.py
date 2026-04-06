"""
orchestrator.py — Orquestrador do pipeline de indexação

Responsabilidades:
  1. Conecta ao Google Drive
  2. Lista arquivos em cada pasta (por agente)
  3. Para cada arquivo PDF, chama pipeline.process_pdf()
  4. Suporta filtro por pasta (folder_filter)
  5. Retorna relatório detalhado da execução

Chamado por:
  - main.py → _run_indexing_job()  (scheduler automático + endpoint /index)
  - Pode ser executado diretamente: python orchestrator.py

Mapeamento pasta Drive → tabela Supabase vem de settings.FOLDER_TABLE_MAP.
"""

import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from gdrive import _get_service, _get_or_create_folder, list_files_in_folder, download_file_bytes  # ← flat import
from pipeline import process_pdf, index_from_json   # ← flat import
from settings import settings           # ← flat import
from downloader import download_public_sources  # ← flat import


# ── Tipos de arquivo suportados ───────────────────────────────────────────────
_SUPPORTED_MIME = {
    "application/pdf",
    "application/octet-stream",   # alguns drives enviam PDF como octet-stream
}
_SUPPORTED_EXT = {".pdf"}


def _is_processable(file: dict) -> bool:
    """Retorna True se o arquivo deve ser processado pelo pipeline."""
    name = file.get("name", "")
    mime = file.get("mimeType", "")
    ext  = Path(name).suffix.lower()
    return ext in _SUPPORTED_EXT or mime in _SUPPORTED_MIME


# ── Ponto de entrada principal ────────────────────────────────────────────────

async def run_indexing(folder_filter: Optional[str] = None) -> dict:
    """
    Percorre as pastas do Google Drive, baixa os PDFs e os indexa no Supabase
    via pipeline.process_pdf().

    Args:
        folder_filter: se informado, processa apenas a pasta com esse nome
                       (ex: "fiscal", "contabil"). None = todas as pastas.

    Returns:
        Relatório com contadores e detalhes por arquivo.
    """
    started_at = datetime.utcnow().isoformat()
    logger.info("=" * 60)
    logger.info(f"▶ Iniciando indexação — {started_at}")
    if folder_filter:
        logger.info(f"  Filtro de pasta: '{folder_filter}'")

    folder_table_map = settings.get_folder_table_map()
    if not folder_table_map:
        logger.error("FOLDER_TABLE_MAP não configurado ou vazio.")
        return {"status": "error", "message": "FOLDER_TABLE_MAP não configurado"}

    svc = _get_service()
    root_folder_id = settings.GDRIVE_ROOT_FOLDER_ID

    report = {
        "started_at":  started_at,
        "folder_filter": folder_filter or "todas",
        "folders":     {},
        "totals": {
            "processed": 0,
            "skipped":   0,
            "error":     0,
            "total_files": 0,
        },
    }

    # ── Passo 1: Download direto (Planalto, CFC, RFB etc.) ───────────────────
    logger.info("📥 Iniciando download de fontes diretas (downloader)...")
    try:
        dl_results  = download_public_sources()
        dl_uploaded = sum(1 for r in dl_results if r.get("status") == "uploaded")
        dl_skipped  = sum(1 for r in dl_results if r.get("status") == "skipped")
        dl_errors   = sum(1 for r in dl_results if r.get("status") == "error")
        logger.info(f"  ✓ Downloader: {dl_uploaded} novos | {dl_skipped} pulados | {dl_errors} erros")
        report["download"] = {"uploaded": dl_uploaded, "skipped": dl_skipped, "errors": dl_errors}
    except Exception as e:
        logger.error(f"  ✗ Downloader falhou: {e}")
        report["download"] = {"error": str(e)}

    # ── Passo 2: Crawler (SEFAZ-MA, DREI, CPC, eSocial etc.) ─────────────────
    logger.info("🌐 Iniciando crawler de portais dinâmicos...")
    try:
        from crawler import run_crawler
        crawl_results = await run_crawler(source_filter=folder_filter)
        cr_uploaded = sum(1 for r in crawl_results if r.get("status") == "uploaded")
        cr_skipped  = sum(1 for r in crawl_results if r.get("status") == "skipped")
        cr_errors   = sum(1 for r in crawl_results if r.get("status") == "error")
        logger.info(f"  ✓ Crawler: {cr_uploaded} novos | {cr_skipped} pulados | {cr_errors} erros")
        report["crawler"] = {"uploaded": cr_uploaded, "skipped": cr_skipped, "errors": cr_errors}
    except Exception as e:
        logger.error(f"  ✗ Crawler falhou: {e}")
        report["crawler"] = {"error": str(e)}

    for folder_name, table_name in folder_table_map.items():

        # ── Aplica filtro de pasta ────────────────────────────────────────────
        if folder_filter and folder_name != folder_filter:
            continue

        logger.info(f"\n📂 Pasta: '{folder_name}' → tabela '{table_name}'")

        folder_report = {
            "table":     table_name,
            "files":     [],
            "processed": 0,
            "skipped":   0,
            "error":     0,
        }

        # ── Resolve folder_id no Drive ────────────────────────────────────────
        try:
            folder_id = _get_or_create_folder(svc, folder_name, root_folder_id)
        except Exception as e:
            logger.error(f"  ✗ Erro ao acessar pasta '{folder_name}': {e}")
            folder_report["error_msg"] = str(e)
            report["folders"][folder_name] = folder_report
            continue

        # ── Lista arquivos ────────────────────────────────────────────────────
        try:
            files = list_files_in_folder(svc, folder_id)
        except Exception as e:
            logger.error(f"  ✗ Erro ao listar arquivos de '{folder_name}': {e}")
            folder_report["error_msg"] = str(e)
            report["folders"][folder_name] = folder_report
            continue

        processable = [f for f in files if _is_processable(f)]
        logger.info(f"  {len(processable)} arquivo(s) para processar de {len(files)} total")
        report["totals"]["total_files"] += len(processable)

        # ── Processa cada arquivo ─────────────────────────────────────────────
        with tempfile.TemporaryDirectory(prefix="fintax_") as tmp_dir:
            for file in processable:
                file_id   = file["id"]
                file_name = file["name"]
                modified  = file.get("modifiedTime", "")

                logger.info(f"  ↓ {file_name}")

                # 1. Download do Drive
                try:
                    pdf_bytes = download_file_bytes(svc, file_id)
                except Exception as e:
                    logger.error(f"    ✗ Download falhou: {e}")
                    folder_report["files"].append({
                        "file": file_name, "status": "error", "error": str(e)
                    })
                    folder_report["error"] += 1
                    report["totals"]["error"] += 1
                    continue

                # 2. Salva temporariamente em disco para o Docling
                tmp_path = Path(tmp_dir) / file_name
                tmp_path.write_bytes(pdf_bytes)

                # 3. FASE 1: PDF → Docling → JSON estruturado → Drive
                try:
                    result_f1 = process_pdf(
                        pdf_path    = tmp_path,
                        file_name   = file_name,
                        file_id     = file_id,
                        folder_name = folder_name,
                        table_name  = table_name,
                        modified_at = modified,
                    )
                except Exception as e:
                    logger.error(f"    ✗ Fase 1 falhou para '{file_name}': {e}")
                    result_f1 = {"status": "error", "file": file_name, "error": str(e)}

                # 4. FASE 2: JSON do Drive → Embeddings → Supabase
                if result_f1.get("status") in ("json_saved", "json_exists"):
                    json_filename = result_f1.get("json_file") or file_name.rsplit(".", 1)[0] + ".json"
                    try:
                        result = index_from_json(
                            json_filename = json_filename,
                            folder_name   = folder_name,
                            table_name    = table_name,
                        )
                    except Exception as e:
                        logger.error(f"    ✗ Fase 2 falhou para '{json_filename}': {e}")
                        result = {"status": "error", "file": file_name, "error": str(e)}
                else:
                    result = result_f1

                # 4. Acumula no relatório
                folder_report["files"].append(result)
                status = result.get("status", "error")
                if status in ("ok", "indexed"):
                    folder_report["processed"] += 1
                    report["totals"]["processed"] += 1
                elif status in ("skipped", "json_exists", "already_indexed"):
                    folder_report["skipped"] += 1
                    report["totals"]["skipped"] += 1
                elif status == "json_saved":
                    # JSON gerado mas indexação falhou — conta como processado parcial
                    folder_report["processed"] += 1
                    report["totals"]["processed"] += 1
                else:
                    folder_report["error"] += 1
                    report["totals"]["error"] += 1

        report["folders"][folder_name] = folder_report

    # ── Sumário final ─────────────────────────────────────────────────────────
    report["finished_at"] = datetime.utcnow().isoformat()
    t = report["totals"]
    logger.info(
        f"\n🏁 Indexação concluída | "
        f"{t['processed']} processados | "
        f"{t['skipped']} pulados | "
        f"{t['error']} erros | "
        f"total {t['total_files']} arquivos"
    )
    return report


# ── Execução direta (debug/teste) ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else None
    result = run_indexing(folder_filter=folder)
    import json
    print(json.dumps(result, indent=2, ensure_ascii=False))
