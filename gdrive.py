"""
gdrive.py — Integração com Google Drive
Compatível com Shared Drives (supportsAllDrives=True em todas as chamadas).
"""

import io
import json
from functools import lru_cache
from typing import Optional

from loguru import logger

from settings import settings  # ← flat import


@lru_cache(maxsize=1)
def _get_service():
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    try:
        creds_dict = json.loads(settings.GOOGLE_CREDENTIALS_JSON)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"GOOGLE_CREDENTIALS_JSON inválido: {e}")
    scopes = ["https://www.googleapis.com/auth/drive"]
    credentials = service_account.Credentials.from_service_account_info(creds_dict, scopes=scopes)
    service = build("drive", "v3", credentials=credentials, cache_discovery=False)
    logger.debug("Google Drive API autenticado com sucesso")
    return service


def _get_or_create_folder(svc, folder_name: str, parent_id: str) -> str:
    query = (
        f"name='{folder_name}' "
        f"and mimeType='application/vnd.google-apps.folder' "
        f"and '{parent_id}' in parents "
        f"and trashed=false"
    )
    resp = svc.files().list(
        q=query, spaces="drive", fields="files(id, name)", pageSize=1,
        supportsAllDrives=True, includeItemsFromAllDrives=True,
    ).execute()
    files = resp.get("files", [])
    if files:
        folder_id = files[0]["id"]
        logger.debug(f"Pasta '{folder_name}' encontrada: {folder_id}")
        return folder_id
    metadata = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }
    folder = svc.files().create(body=metadata, fields="id", supportsAllDrives=True).execute()
    folder_id = folder["id"]
    logger.info(f"Pasta '{folder_name}' criada no Drive: {folder_id}")
    return folder_id


def _pdf_exists_in_folder(svc, filename: str, folder_id: str) -> bool:
    safe_name = filename.replace("'", "\\'")
    query = f"name='{safe_name}' and '{folder_id}' in parents and trashed=false"
    resp = svc.files().list(
        q=query, spaces="drive", fields="files(id)", pageSize=1,
        supportsAllDrives=True, includeItemsFromAllDrives=True,
    ).execute()
    return len(resp.get("files", [])) > 0


def _get_file_id_in_folder(svc, filename: str, folder_id: str) -> Optional[str]:
    safe_name = filename.replace("'", "\\'")
    query = f"name='{safe_name}' and '{folder_id}' in parents and trashed=false"
    resp = svc.files().list(
        q=query, spaces="drive", fields="files(id)", pageSize=1,
        supportsAllDrives=True, includeItemsFromAllDrives=True,
    ).execute()
    files = resp.get("files", [])
    return files[0]["id"] if files else None


def _upload_bytes_to_drive(svc, content: bytes, filename: str, folder_id: str, mime_type: str = "application/pdf") -> str:
    from googleapiclient.http import MediaIoBaseUpload
    file_metadata = {"name": filename, "parents": [folder_id]}
    media = MediaIoBaseUpload(io.BytesIO(content), mimetype=mime_type, resumable=True)
    created = svc.files().create(
        body=file_metadata, media_body=media, fields="id", supportsAllDrives=True,
    ).execute()
    file_id = created["id"]
    logger.debug(f"Upload concluído: '{filename}' → Drive {file_id}")
    return file_id


def download_file_bytes(svc, file_id: str) -> bytes:
    from googleapiclient.http import MediaIoBaseDownload
    request = svc.files().get_media(fileId=file_id, supportsAllDrives=True)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buf.getvalue()


def list_files_in_folder(svc, folder_id: str, page_size: int = 200) -> list[dict]:
    results = []
    page_token = None
    while True:
        query = (
            f"'{folder_id}' in parents "
            f"and mimeType != 'application/vnd.google-apps.folder' "
            f"and trashed=false"
        )
        kwargs = dict(
            q=query, spaces="drive",
            fields="nextPageToken, files(id, name, mimeType, modifiedTime, size)",
            pageSize=page_size,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        )
        if page_token:
            kwargs["pageToken"] = page_token
        resp = svc.files().list(**kwargs).execute()
        results.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return results
