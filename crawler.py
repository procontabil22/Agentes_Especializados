"""
app/crawler.py
Crawler web para portais governamentais — 5 agentes FinTax.

4 camadas de proteção contra re-download:
  Camada 1 — URL Hash (Supabase crawl_log): nunca baixa a mesma URL duas vezes
  Camada 2 — Nome no Drive: não sobe arquivo com nome duplicado
  Camada 3 — Content Hash (SHA-256): detecta mesmo PDF em URL diferente
  Camada 4 — Pipeline Supabase: não reprocessa embeddings de arquivo já indexado

Padrão SEFAZ-MA (portais JSF):
  Páginas: /portalsefaz/jsp/pagina/pagina.jsf?codigo=NUM
  PDFs:    /portalsefaz/pdf?codigo=NUM
"""
import asyncio, hashlib, re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx
from loguru import logger
from supabase import create_client

from app.gdrive import _get_service, _get_or_create_folder, _pdf_exists_in_folder, _upload_bytes_to_drive
from config.settings import settings

_supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)


# ── Modelo de fonte ───────────────────────────────────────────────────────────
@dataclass
class CrawlSource:
    url: str
    folder_name: str
    description: str = ""
    max_depth: int = 2
    pdf_pattern: str = r"\.pdf"
    direct_pdf_pattern: str = ""
    same_domain_only: bool = True
    use_browser: bool = False
    headers: dict = field(default_factory=dict)


# ════════════════════════════════════════════════════════════════════════════
# FONTES A VARRER POR AGENTE
# Adicione novas URLs aqui — o crawler extrai todos os PDFs automaticamente
# ════════════════════════════════════════════════════════════════════════════
CRAWL_SOURCES: list[CrawlSource] = [

    # ── FISCAL — SEFAZ-MA (portais JSF — requer Playwright) ──────────────────
    CrawlSource(
        url="https://sistemas1.sefaz.ma.gov.br/portalsefaz/jsp/pagina/pagina.jsf?codigo=95",
        folder_name="fiscal",
        description="SEFAZ-MA — Legislação ICMS (RICMS-MA e Decretos)",
        use_browser=True,
        direct_pdf_pattern=r"/portalsefaz/pdf\?codigo=\d+",
        max_depth=2,
    ),
    CrawlSource(
        url="https://sistemas1.sefaz.ma.gov.br/portalsefaz/jsp/pagina/pagina.jsf?codigo=98",
        folder_name="fiscal",
        description="SEFAZ-MA — Anexos do RICMS-MA",
        use_browser=True,
        direct_pdf_pattern=r"/portalsefaz/pdf\?codigo=\d+",
        max_depth=2,
    ),
    CrawlSource(
        url="https://sistemas1.sefaz.ma.gov.br/portalsefaz/jsp/pagina/pagina.jsf?codigo=3556",
        folder_name="fiscal",
        description="SEFAZ-MA — Benefícios Fiscais e Desonerações ICMS/IPVA",
        use_browser=True,
        direct_pdf_pattern=r"/portalsefaz/pdf\?codigo=\d+",
        max_depth=2,
    ),
    CrawlSource(
        url="https://sistemas1.sefaz.ma.gov.br/portalsefaz/jsp/pagina/pagina.jsf?codigo=97",
        folder_name="fiscal",
        description="SEFAZ-MA — Substituição Tributária MA",
        use_browser=True,
        direct_pdf_pattern=r"/portalsefaz/pdf\?codigo=\d+",
        max_depth=2,
    ),

    # ── FISCAL — Receita Federal (site estático) ──────────────────────────────
    CrawlSource(
        url="https://www.planalto.gov.br/ccivil_03/leis/lcp/lcp214.htm",
        folder_name="fiscal",
        description="LC 214/2024 — Reforma Tributária",
        use_browser=False, max_depth=0,
    ),

    # ── CONTÁBIL — CPC (site estático com links PDF) ──────────────────────────
    CrawlSource(
        url="https://www.cpc.org.br/CPC/Documentos-Emitidos/Pronunciamentos",
        folder_name="contabil",
        description="CPC — Pronunciamentos Contábeis (todos os CPCs em PDF)",
        use_browser=False,
        pdf_pattern=r"\.pdf",
        max_depth=1,
        same_domain_only=True,
    ),

    # ── PESSOAL — Planalto ────────────────────────────────────────────────────
    CrawlSource(
        url="https://www.planalto.gov.br/ccivil_03/decreto-lei/del5452.htm",
        folder_name="pessoal",
        description="CLT compilada — Planalto",
        use_browser=False, max_depth=0,
    ),
    CrawlSource(
        url="https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2017/lei/l13467.htm",
        folder_name="pessoal",
        description="Lei 13.467/2017 — Reforma Trabalhista",
        use_browser=False, max_depth=0,
    ),

    # ── SOCIETÁRIO — Planalto ─────────────────────────────────────────────────
    CrawlSource(
        url="https://www.planalto.gov.br/ccivil_03/leis/2002/l10406compilada.htm",
        folder_name="societario",
        description="Código Civil 2002 compilado",
        use_browser=False, max_depth=0,
    ),
    CrawlSource(
        url="https://www.planalto.gov.br/ccivil_03/leis/l6404consol.htm",
        folder_name="societario",
        description="Lei 6.404/1976 — Lei das S.A.",
        use_browser=False, max_depth=0,
    ),

    # ── ABERTURA MA — CBMMA (se tiver página pública com ITs) ────────────────
    # Exemplo: se o CBMMA publicar as ITs em site estático
    # CrawlSource(
    #     url="https://www.cbm.ma.gov.br/instrucoes-tecnicas/",
    #     folder_name="abertura_ma",
    #     description="CBMMA — Instruções Técnicas de Segurança Contra Incêndio",
    #     use_browser=False,
    #     pdf_pattern=r"\.pdf",
    #     max_depth=1,
    # ),
]


# ── Funções de deduplicação ───────────────────────────────────────────────────

def _hash_url(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()

def _hash_content(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def _url_already_downloaded(url: str) -> Optional[dict]:
    """CAMADA 1: URL hash no Supabase crawl_log."""
    resp = (
        _supabase.table("crawl_log")
        .select("id,filename,downloaded_at,drive_file_id")
        .eq("url_hash", _hash_url(url))
        .eq("status", "downloaded")
        .limit(1).execute()
    )
    return resp.data[0] if resp.data else None

def _content_already_exists(content_hash: str) -> Optional[dict]:
    """CAMADA 3: mesmo PDF em URL diferente."""
    resp = (
        _supabase.table("crawl_log")
        .select("id,filename,url,drive_file_id")
        .eq("content_hash", content_hash)
        .eq("status", "downloaded")
        .limit(1).execute()
    )
    return resp.data[0] if resp.data else None

def _log_crawl(url, filename, folder, status, source_page="",
               drive_file_id="", size_kb=0, content_hash="", error_msg=""):
    now = datetime.now(timezone.utc).isoformat()
    record = {
        "url": url, "url_hash": _hash_url(url),
        "filename": filename, "folder_name": folder,
        "status": status, "source_page": source_page,
        "last_checked_at": now,
    }
    if drive_file_id: record["drive_file_id"] = drive_file_id
    if size_kb:       record["file_size_kb"] = size_kb
    if content_hash:  record["content_hash"] = content_hash
    if error_msg:     record["error_msg"] = error_msg
    if status == "downloaded": record["downloaded_at"] = now
    _supabase.table("crawl_log").upsert(record, on_conflict="url_hash").execute()


# ── Conversão URL → filename ──────────────────────────────────────────────────
def _url_to_filename(url: str, title: str = "") -> str:
    m = re.search(r"codigo=(\d+)", url)
    if m:
        code = m.group(1)
        safe = re.sub(r"[^\w\-]", "_", title)[:50] if title else f"doc_{code}"
        return f"SEFAZMA_{safe}_{code}.pdf"
    path = urlparse(url).path
    name = path.rstrip("/").split("/")[-1]
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return re.sub(r"[^\w\-\.]", "_", name)


# ── Busca de página ───────────────────────────────────────────────────────────
async def _fetch_httpx(url: str, headers: dict = None) -> str:
    async with httpx.AsyncClient(
        timeout=30, follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 (compatible; FinTaxBot/1.0)", **(headers or {})}
    ) as c:
        r = await c.get(url)
        r.raise_for_status()
        return r.text

async def _fetch_playwright(url: str) -> str:
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        logger.error("Playwright não instalado. Adicione ao requirements.txt")
        return ""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.set_extra_http_headers({"User-Agent": "Mozilla/5.0 (compatible; FinTaxBot/1.0)"})
        try:
            await page.goto(url, wait_until="networkidle", timeout=45000)
            await asyncio.sleep(2)
            html = await page.content()
        finally:
            await browser.close()
        return html


# ── Extração de links de PDF ──────────────────────────────────────────────────
def _extract_pdf_links(html: str, base_url: str, source: CrawlSource) -> list[dict]:
    found, seen = [], set()
    base_domain = urlparse(base_url).netloc

    # Padrão direto (ex: pdf?codigo=)
    if source.direct_pdf_pattern:
        for m in re.finditer(source.direct_pdf_pattern, html):
            full = urljoin(base_url, m.group(0))
            if full not in seen:
                seen.add(full)
                found.append({"url": full, "title": ""})

    # Tags <a href>
    for m in re.finditer(r'<a\s[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
                         html, re.IGNORECASE | re.DOTALL):
        href = m.group(1)
        text = re.sub(r"<[^>]+>", "", m.group(2)).strip()
        full = urljoin(base_url, href)

        if source.same_domain_only and urlparse(full).netloc != base_domain:
            continue

        is_pdf = (re.search(source.pdf_pattern, full, re.IGNORECASE)
                  or (source.direct_pdf_pattern and re.search(source.direct_pdf_pattern, full)))

        if is_pdf and full not in seen:
            seen.add(full)
            found.append({"url": full, "title": text[:100]})

    return found

def _extract_sub_links(html: str, base_url: str) -> list[str]:
    links, seen = [], set()
    base_domain = urlparse(base_url).netloc
    for m in re.finditer(r'href=["\']([^"\'#]+)["\']', html, re.IGNORECASE):
        href = m.group(1)
        full = urljoin(base_url, href)
        if urlparse(full).netloc != base_domain: continue
        if full in seen or full == base_url: continue
        if re.search(r"\.(pdf|doc|docx|xls|zip)$", full, re.IGNORECASE): continue
        seen.add(full)
        links.append(full)
    return links[:20]


# ── Download do PDF ───────────────────────────────────────────────────────────
async def _download_pdf(url: str) -> Optional[bytes]:
    try:
        async with httpx.AsyncClient(timeout=60, follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; FinTaxBot/1.0)"}) as c:
            r = await c.get(url)
            r.raise_for_status()
            if not (r.content[:4] == b"%PDF" or "pdf" in r.headers.get("content-type","")):
                if ".pdf" not in url.lower():
                    logger.warning(f"  ⚠ Não é PDF: {url}")
                    return None
            return r.content
    except Exception as e:
        logger.error(f"  ✗ Download {url}: {e}")
        return None


# ── Orquestra o crawl de uma fonte ───────────────────────────────────────────
async def crawl_and_upload(source: CrawlSource) -> list[dict]:
    logger.info(f"\n🌐 Varrendo: {source.description or source.url}")
    svc = _get_service()
    folder_id = _get_or_create_folder(svc, source.folder_name, settings.GDRIVE_ROOT_FOLDER_ID)

    visited, pdf_links = set(), []

    async def crawl_page(url: str, depth: int):
        if url in visited or depth > source.max_depth:
            return
        visited.add(url)
        try:
            html = await (_fetch_playwright(url) if source.use_browser else _fetch_httpx(url, source.headers))
        except Exception as e:
            logger.error(f"  ✗ Erro página {url}: {e}")
            return
        new = [p for p in _extract_pdf_links(html, url, source)
               if p["url"] not in {x["url"] for x in pdf_links}]
        pdf_links.extend(new)
        if new: logger.info(f"  🔗 {len(new)} PDFs encontrados em {url}")
        if depth < source.max_depth:
            for sub in _extract_sub_links(html, url):
                await crawl_page(sub, depth + 1)
                await asyncio.sleep(0.5)

    await crawl_page(source.url, 0)
    logger.info(f"  Total PDFs encontrados: {len(pdf_links)}")

    results = []
    for item in pdf_links:
        url, title = item["url"], item.get("title", "")
        filename = _url_to_filename(url, title)

        # ── Camada 1: URL hash no crawl_log ──────────────────────────────────
        existing = _url_already_downloaded(url)
        if existing:
            logger.debug(f"  ⏭ [C1-URL] Já baixado: {filename}")
            _log_crawl(url, filename, source.folder_name, "downloaded",
                       source.url, existing.get("drive_file_id",""))
            results.append({"file": filename, "status": "skipped", "reason": "url_in_log"})
            continue

        # ── Camada 2: nome do arquivo no Drive ────────────────────────────────
        if _pdf_exists_in_folder(svc, filename, folder_id):
            logger.debug(f"  ⏭ [C2-Drive] Já existe: {filename}")
            _log_crawl(url, filename, source.folder_name, "downloaded", source.url)
            results.append({"file": filename, "status": "skipped", "reason": "name_in_drive"})
            continue

        # Download
        logger.info(f"  ↓ Baixando: {filename}")
        pdf_bytes = await _download_pdf(url)
        if not pdf_bytes:
            _log_crawl(url, filename, source.folder_name, "error", source.url,
                       error_msg="download falhou")
            results.append({"file": filename, "status": "error", "url": url})
            continue

        # ── Camada 3: hash do conteúdo ────────────────────────────────────────
        c_hash = _hash_content(pdf_bytes)
        dup = _content_already_exists(c_hash)
        if dup:
            logger.debug(f"  ⏭ [C3-Hash] Mesmo conteúdo de '{dup['filename']}'")
            _log_crawl(url, filename, source.folder_name, "downloaded", source.url,
                       dup.get("drive_file_id",""), len(pdf_bytes)//1024, c_hash)
            results.append({"file": filename, "status": "skipped",
                            "reason": "content_duplicate", "original": dup["filename"]})
            continue

        # Upload para o Drive
        try:
            drive_id = _upload_bytes_to_drive(svc, pdf_bytes, filename, folder_id)
            size_kb = len(pdf_bytes) // 1024
            logger.success(f"  ✓ Upload: {filename} ({size_kb}KB)")
            _log_crawl(url, filename, source.folder_name, "downloaded",
                       source.url, drive_id, size_kb, c_hash)
            results.append({"file": filename, "status": "uploaded",
                            "drive_id": drive_id, "size_kb": size_kb})
        except Exception as e:
            logger.error(f"  ✗ Upload falhou: {e}")
            _log_crawl(url, filename, source.folder_name, "error", source.url, error_msg=str(e))
            results.append({"file": filename, "status": "error", "error": str(e)})

        await asyncio.sleep(1)

    return results


async def run_crawler(source_filter: str | None = None) -> list[dict]:
    all_results = []
    for source in CRAWL_SOURCES:
        if source_filter and source.folder_name != source_filter:
            continue
        all_results.extend(await crawl_and_upload(source))

    up  = sum(1 for r in all_results if r["status"] == "uploaded")
    sk  = sum(1 for r in all_results if r["status"] == "skipped")
    err = sum(1 for r in all_results if r["status"] == "error")
    logger.info(f"\n🏁 Crawler: {up} uploads | {sk} pulados | {err} erros")
    return all_results
