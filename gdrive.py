"""
gdrive.py — Integração com Google Drive

Funções utilizadas por downloader.py e crawler.py:
  _get_service()              → autentica e retorna o cliente Drive
  _get_or_create_folder()     → retorna folder_id (cria se não existir)
  _pdf_exists_in_folder()     → verifica se arquivo já existe pelo nome
  _upload_bytes_to_drive()    → faz upload de bytes e retorna file_id

Autenticação via GOOGLE_CREDENTIALS_JSON (service account JSON como string).
Configure a variável no Railway: Settings > Variables.
"""

import io
import json
from functools import lru_cache
from typing import Optional

from loguru import logger

from settings import settings  # ← flat import


# ── Cliente autenticado ───────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_service():
    """
    Retorna o cliente autenticado do Google Drive API v3.
    Usa Service Account — o JSON completo fica em GOOGLE_CREDENTIALS_JSON.
    """
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    try:
        creds_dict = json.loads(settings.GOOGLE_CREDENTIALS_JSON)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"GOOGLE_CREDENTIALS_JSON inválido (não é JSON válido): {e}"
        )

    scopes = ["https://www.googleapis.com/auth/drive"]
    credentials = service_account.Credentials.from_service_account_info(
        creds_dict, scopes=scopes
    )
    service = build("drive", "v3", credentials=credentials, cache_discovery=False)
    logger.debug("Google Drive API autenticado com sucesso")
    return service


# ── Gerenciamento de pastas ───────────────────────────────────────────────────

def _get_or_create_folder(svc, folder_name: str, parent_id: str) -> str:
    """
    Retorna o ID da pasta `folder_name` dentro de `parent_id`.
    Se não existir, cria e retorna o novo ID.
    """
    query = (
        f"name='{folder_name}' "
        f"and mimeType='application/vnd.google-apps.folder' "
        f"and '{parent_id}' in parents "
        f"and trashed=false"
    )
    resp = svc.files().list(
        q=query,
        spaces="drive",
        fields="files(id, name)",
        pageSize=1,
    ).execute()

    files = resp.get("files", [])
    if files:
        folder_id = files[0]["id"]
        logger.debug(f"Pasta '{folder_name}' encontrada: {folder_id}")
        return folder_id

    # Cria a pasta
    metadata = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }
    folder = svc.files().create(body=metadata, fields="id").execute()
    folder_id = folder["id"]
    logger.info(f"Pasta '{folder_name}' criada no Drive: {folder_id}")
    return folder_id


# ── Verificação de existência ─────────────────────────────────────────────────

def _pdf_exists_in_folder(svc, filename: str, folder_id: str) -> bool:
    """
    Retorna True se já existe um arquivo com exatamente esse nome
    dentro da pasta `folder_id` (não na lixeira).
    """
    # Escapa aspas simples no nome para não quebrar a query
    safe_name = filename.replace("'", "\\'")
    query = (
        f"name='{safe_name}' "
        f"and '{folder_id}' in parents "
        f"and trashed=false"
    )
    resp = svc.files().list(
        q=query,
        spaces="drive",
        fields="files(id)",
        pageSize=1,
    ).execute()
    return len(resp.get("files", [])) > 0


def _get_file_id_in_folder(svc, filename: str, folder_id: str) -> Optional[str]:
    """
    Retorna o file_id se o arquivo existir, None caso contrário.
    """
    safe_name = filename.replace("'", "\\'")
    query = (
        f"name='{safe_name}' "
        f"and '{folder_id}' in parents "
        f"and trashed=false"
    )
    resp = svc.files().list(
        q=query,
        spaces="drive",
        fields="files(id)",
        pageSize=1,
    ).execute()
    files = resp.get("files", [])
    return files[0]["id"] if files else None


# ── Upload ────────────────────────────────────────────────────────────────────

def _upload_bytes_to_drive(
    svc,
    content: bytes,
    filename: str,
    folder_id: str,
    mime_type: str = "application/pdf",
) -> str:
    """
    Faz upload de `content` (bytes) para o Drive na pasta `folder_id`.
    Retorna o file_id do arquivo criado.

    Usa resumable upload via MediaIoBaseUpload para suportar arquivos grandes.
    """
    from googleapiclient.http import MediaIoBaseUpload

    file_metadata = {
        "name": filename,
        "parents": [folder_id],
    }

    media = MediaIoBaseUpload(
        io.BytesIO(content),
        mimetype=mime_type,
        resumable=True,
    )

    created = svc.files().create(
        body=file_metadata,
        media_body=media,
        fields="id",
    ).execute()

    file_id = created["id"]
    logger.debug(f"Upload concluído: '{filename}' → Drive {file_id}")
    return file_id


# ── Download (usado pelo pipeline para processar PDFs do Drive) ───────────────

def download_file_bytes(svc, file_id: str) -> bytes:
    """
    Faz download do conteúdo de um arquivo do Drive pelo file_id.
    Retorna os bytes brutos.
    """
    from googleapiclient.http import MediaIoBaseDownload

    request = svc.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buf.getvalue()


def list_files_in_folder(svc, folder_id: str, page_size: int = 200) -> list[dict]:
    """
    Lista todos os arquivos (não pastas) dentro de `folder_id`.
    Retorna lista de dicts com id, name, mimeType, modifiedTime.
    Suporta paginação automática.
    """
    results = []
    page_token = None

    while True:
        query = (
            f"'{folder_id}' in parents "
            f"and mimeType != 'application/vnd.google-apps.folder' "
            f"and trashed=false"
        )
        kwargs = dict(
            q=query,
            spaces="drive",
            fields="nextPageToken, files(id, name, mimeType, modifiedTime, size)",
            pageSize=page_size,
        )
        if page_token:
            kwargs["pageToken"] = page_token

        resp = svc.files().list(**kwargs).execute()
        results.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    return results
