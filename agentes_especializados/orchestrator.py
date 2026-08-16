"""
orchestrator.py — Orquestrador Multi-Formato v2.0

Melhorias:
  ✓ Suporta PDF, DOCX, XLSX, PPTX, EML, MSG, PNG, JPEG, TIFF
  ✓ Fila por prioridade: imagens/emails primeiro, PDFs grandes por último
  ✓ Contadores por formato no relatório
"""

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from gdrive import (_get_service, _get_or_create_folder,
                    list_files_in_folder, download_file_bytes)
from pipeline import process_document, index_from_json
from settings import settings
from downloader import download_public_sources


# ── Formatos suportados ───────────────────────────────────────────────────────
_SUPPORTED_EXT = {
    # Documentos
    ".pdf", ".docx", ".doc", ".pptx", ".ppt",
    # Planilhas
    ".xlsx", ".xls",
    # Email
    ".eml", ".msg",
    # Imagens
    ".png", ".jpg", ".jpeg", ".tiff", ".tif",
}

_SUPPORTED_MIME = {
    "application/pdf",
    "application/octet-stream",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "message/rfc822",
    "image/png",
    "image/jpeg",
    "image/tiff",
}

# Prioridade de processamento (menor = primeiro)
_EXT_PRIORITY = {
    ".eml": 1, ".msg": 1,         # emails: rápidos
    ".png": 2, ".jpg": 2,
    ".jpeg": 2, ".tiff": 2,       # imagens: OCR rápido
    ".xlsx": 3, ".xls": 3,        # planilhas: médio
    ".docx": 3, ".doc": 3,
    ".pptx": 4, ".ppt": 4,
    ".pdf": 5,                     # PDFs: mais pesados
}


def _is_processable(file: dict) -> bool:
    name = file.get("name", "")
    mime = file.get("mimeType", "")
    ext  = Path(name).suffix.lower()
    # Pula JSONs gerados pelo pipeline
    if ext == ".json":
        return False
    return ext in _SUPPORTED_EXT or mime in _SUPPORTED_MIME


def _sort_key(file: dict) -> tuple:
    """Ordena por prioridade e depois por tamanho (menor primeiro)."""
    ext  = Path(file.get("name", "")).suffix.lower()
    size = int(file.get("size", 0))
    prio = _EXT_PRIORITY.get(ext, 5)
    return (prio, size)


async def run_indexing(folder_filter: Optional[str] = None) -> dict:
    started_at = datetime.utcnow().isoformat()
    logger.info("=" * 60)
    logger.info(f"▶ Iniciando indexação v2.0 — {started_at}")
    if folder_filter:
        logger.info(f"  Filtro: '{folder_filter}'")

    folder_table_map = settings.get_folder_table_map()
    if not folder_table_map:
        return {"status": "error", "message": "FOLDER_TABLE_MAP não configurado"}

    svc            = _get_service()
    root_folder_id = settings.GDRIVE_ROOT_FOLDER_ID

    report = {
        "started_at":    started_at,
        "folder_filter": folder_filter or "todas",
        "folders":       {},
        "totals": {
            "processed": 0, "skipped": 0,
            "error": 0, "total_files": 0,
        },
        "formats": {},  # contadores por extensão
    }

    # ── Download fontes públicas ──────────────────────────────────────────────
    logger.info("📥 Downloader...")
    try:
        dl = download_public_sources()
        dl_up  = sum(1 for r in dl if r.get("status") == "uploaded")
        dl_sk  = sum(1 for r in dl if r.get("status") == "skipped")
        dl_err = sum(1 for r in dl if r.get("status") == "error")
        logger.info(f"  ✓ {dl_up} novos | {dl_sk} pulados | {dl_err} erros")
        report["download"] = {"uploaded": dl_up, "skipped": dl_sk, "errors": dl_err}
    except Exception as e:
        logger.error(f"  ✗ Downloader: {e}")
        report["download"] = {"error": str(e)}

    # ── Crawler ───────────────────────────────────────────────────────────────
    logger.info("🌐 Crawler...")
    try:
        from crawler import run_crawler
        cr = await run_crawler(source_filter=folder_filter)
        cr_up  = sum(1 for r in cr if r.get("status") == "uploaded")
        cr_sk  = sum(1 for r in cr if r.get("status") == "skipped")
        cr_err = sum(1 for r in cr if r.get("status") == "error")
        logger.info(f"  ✓ {cr_up} novos | {cr_sk} pulados | {cr_err} erros")
        report["crawler"] = {"uploaded": cr_up, "skipped": cr_sk, "errors": cr_err}
    except Exception as e:
        logger.error(f"  ✗ Crawler: {e}")
        report["crawler"] = {"error": str(e)}

    # ── Processa cada pasta ───────────────────────────────────────────────────
    for folder_name, table_name in folder_table_map.items():
        if folder_filter and folder_name != folder_filter:
            continue

        logger.info(f"\n📂 '{folder_name}' → '{table_name}'")

        folder_report = {
            "table": table_name, "files": [],
            "processed": 0, "skipped": 0, "error": 0,
        }

        try:
            folder_id = _get_or_create_folder(svc, folder_name, root_folder_id)
        except Exception as e:
            logger.error(f"  ✗ Pasta '{folder_name}': {e}")
            folder_report["error_msg"] = str(e)
            report["folders"][folder_name] = folder_report
            continue

        try:
            files = list_files_in_folder(svc, folder_id)
        except Exception as e:
            logger.error(f"  ✗ Listar '{folder_name}': {e}")
            folder_report["error_msg"] = str(e)
            report["folders"][folder_name] = folder_report
            continue

        processable = sorted(
            [f for f in files if _is_processable(f)],
            key=_sort_key
        )
        logger.info(f"  {len(processable)} arquivo(s) ({len(files)} total)")
        report["totals"]["total_files"] += len(processable)

        with tempfile.TemporaryDirectory(prefix="fintax_") as tmp_dir:
            for file in processable:
                file_id   = file["id"]
                file_name = file["name"]
                modified  = file.get("modifiedTime", "")
                ext       = Path(file_name).suffix.lower()

                # Contador por formato
                report["formats"][ext] = report["formats"].get(ext, 0) + 1

                logger.info(f"  ↓ {file_name}")

                try:
                    file_bytes = download_file_bytes(svc, file_id)
                except Exception as e:
                    logger.error(f"    ✗ Download: {e}")
                    folder_report["files"].append(
                        {"file": file_name, "status": "error", "error": str(e)})
                    folder_report["error"] += 1
                    report["totals"]["error"] += 1
                    continue

                tmp_path = Path(tmp_dir) / file_name
                tmp_path.write_bytes(file_bytes)

                # FASE 1
                try:
                    result_f1 = process_document(
                        file_path   = tmp_path,
                        file_name   = file_name,
                        file_id     = file_id,
                        folder_name = folder_name,
                        table_name  = table_name,
                        modified_at = modified,
                    )
                except Exception as e:
                    logger.error(f"    ✗ Fase 1 '{file_name}': {e}")
                    result_f1 = {"status": "error", "file": file_name, "error": str(e)}

                # FASE 2
                if result_f1.get("status") in ("json_saved", "json_exists"):
                    json_filename = (result_f1.get("json_file")
                                     or Path(file_name).stem + ".json")
                    try:
                        result = index_from_json(
                            json_filename = json_filename,
                            folder_name   = folder_name,
                            table_name    = table_name,
                        )
                    except Exception as e:
                        logger.error(f"    ✗ Fase 2 '{json_filename}': {e}")
                        result = {"status": "error", "file": file_name, "error": str(e)}
                else:
                    result = result_f1

                folder_report["files"].append(result)
                status = result.get("status", "error")
                if status in ("ok", "indexed"):
                    folder_report["processed"] += 1
                    report["totals"]["processed"] += 1
                elif status in ("skipped", "json_exists", "already_indexed"):
                    folder_report["skipped"] += 1
                    report["totals"]["skipped"] += 1
                elif status == "json_saved":
                    folder_report["processed"] += 1
                    report["totals"]["processed"] += 1
                else:
                    folder_report["error"] += 1
                    report["totals"]["error"] += 1

        report["folders"][folder_name] = folder_report

    report["finished_at"] = datetime.utcnow().isoformat()
    t = report["totals"]
    logger.info(
        f"\n🏁 Concluído | {t['processed']} processados | "
        f"{t['skipped']} pulados | {t['error']} erros | "
        f"{t['total_files']} total"
    )
    if report["formats"]:
        logger.info(f"  Formatos: {report['formats']}")
    return report


if __name__ == "__main__":
    import sys, asyncio, json as _json
    folder = sys.argv[1] if len(sys.argv) > 1 else None
    result = asyncio.run(run_indexing(folder_filter=folder))
    print(_json.dumps(result, indent=2, ensure_ascii=False))
