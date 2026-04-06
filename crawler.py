"""
crawler.py — Crawler web para portais governamentais.
Usado para fontes que exigem navegação dinâmica (JSF, SPAs) ou
que listam PDFs em índices HTML (CPC, CFC, DREI, eSocial, etc.).

4 camadas de proteção contra re-download:
  C1 — URL hash no crawl_log
  C2 — nome do arquivo no Google Drive
  C3 — hash do conteúdo (dedup de arquivos idênticos)
  C4 — Playwright para portais com JavaScript
"""
import asyncio
import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx
from loguru import logger

from gdrive import _get_service, _get_or_create_folder, _pdf_exists_in_folder, _upload_bytes_to_drive  # flat import
from settings import settings  # flat import


@lru_cache(maxsize=1)
def _supabase():
    from supabase import create_client
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)


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


# =============================================================================
# FONTES DE CRAWLER POR AGENTE
# Use para portais que exigem navegação (JSF, SPA) ou índices com links de PDF.
# Fontes simples de download direto → use downloader.py
# =============================================================================
CRAWL_SOURCES: list[CrawlSource] = [

    # =========================================================================
    # 1. ANALISTA FISCAL — SEFAZ-MA (portal JSF — exige Playwright)
    #
    # Estratégia: iniciar nas páginas-raiz reais (códigos confirmados via
    # Google) e deixar o Playwright descobrir todos os PDFs automaticamente.
    # Padrões de PDF confirmados:
    #   /portalsefaz/pdf?codigo=XXXX   → documentos PDF
    #   /portalsefaz/files?codigo=XXXX → arquivos (PDF/DOC)
    # =========================================================================

    # Página principal de Legislação (contém links para RICMS, Lei 7.799, etc.)
    CrawlSource(
        url="https://sistemas1.sefaz.ma.gov.br/portalsefaz/jsp/pagina/pagina.jsf?codigo=95",
        folder_name="analista_fiscal",
        description="SEFAZ-MA — Legislação Tributária (raiz: RICMS, Lei 7.799, Convênios, Portarias…)",
        use_browser=True,
        direct_pdf_pattern=r"/portalsefaz/(?:pdf|files)\?codigo=\d+",
        max_depth=3,
    ),
    # Anexos do RICMS-MA (código confirmado via Google)
    # Contém: Anexo 1 (isenções), Anexo 4 (ICMS-ST), Anexo 5 (diferimento),
    # redução de BC, gado bovino, combustíveis, etc.
    CrawlSource(
        url="https://sistemas1.sefaz.ma.gov.br/portalsefaz/jsp/pagina/pagina.jsf?codigo=98",
        folder_name="analista_fiscal",
        description="SEFAZ-MA — Anexos do RICMS-MA (isenção, ST, diferimento, redução, gado bovino)",
        use_browser=True,
        direct_pdf_pattern=r"/portalsefaz/(?:pdf|files)\?codigo=\d+",
        max_depth=3,
    ),
    # Benefícios Fiscais e Desonerações — ICMS/IPVA (código confirmado via Google)
    CrawlSource(
        url="https://sistemas1.sefaz.ma.gov.br/portalsefaz/jsp/pagina/pagina.jsf?codigo=3556",
        folder_name="analista_fiscal",
        description="SEFAZ-MA — Benefícios Fiscais ICMS-MA (isenção, redução, diferimento, ICMS-ST)",
        use_browser=True,
        direct_pdf_pattern=r"/portalsefaz/(?:pdf|files)\?codigo=\d+",
        max_depth=3,
    ),
    # Benefícios Fiscais Concedidos pelo Estado (código confirmado via Google)
    CrawlSource(
        url="https://sistemas1.sefaz.ma.gov.br/portalsefaz/jsp/pagina/pagina.jsf?codigo=1548",
        folder_name="analista_fiscal",
        description="SEFAZ-MA — Benefícios Fiscais Concedidos pelo Maranhão (relatórios, programas)",
        use_browser=True,
        direct_pdf_pattern=r"/portalsefaz/(?:pdf|files)\?codigo=\d+",
        max_depth=2,
    ),
    # Portarias SEFAZ-MA (código confirmado via Google)
    CrawlSource(
        url="https://sistemas1.sefaz.ma.gov.br/portalsefaz/jsp/pagina/pagina.jsf?codigo=104",
        folder_name="analista_fiscal",
        description="SEFAZ-MA — Portarias GABIN/GARE",
        use_browser=True,
        direct_pdf_pattern=r"/portalsefaz/(?:pdf|files)\?codigo=\d+",
        max_depth=2,
    ),

    # ── CONFAZ ────────────────────────────────────────────────────────────────
    # Convênios ICMS — índice completo (Conv. 142/2018, 52/2017, 93/2015…)
    CrawlSource(
        url="https://www.confaz.fazenda.gov.br/legislacao/convenios",
        folder_name="analista_fiscal",
        description="CONFAZ — Convênios ICMS (Conv. 142/2018, 52/2017, 93/2015, 100/1997…)",
        use_browser=False,
        pdf_pattern=r"\.pdf",
        direct_pdf_pattern=r"convenios/\d{4}/CV\d+",
        max_depth=2,
        same_domain_only=True,
    ),
    # Protocolos ICMS — ST interestadual, gado bovino, aves, suínos
    CrawlSource(
        url="https://www.confaz.fazenda.gov.br/legislacao/protocolos",
        folder_name="analista_fiscal",
        description="CONFAZ — Protocolos ICMS (ST gado bovino, aves, suínos, combustíveis)",
        use_browser=False,
        pdf_pattern=r"\.pdf",
        direct_pdf_pattern=r"protocolos/\d{4}/PT\d+",
        max_depth=2,
        same_domain_only=True,
    ),
    # Ajustes SINIEF — NF-e, MDF-e, CT-e, EFD
    CrawlSource(
        url="https://www.confaz.fazenda.gov.br/legislacao/ajustes",
        folder_name="analista_fiscal",
        description="CONFAZ — Ajustes SINIEF (NF-e, MDF-e, CT-e, EFD ICMS/IPI)",
        use_browser=False,
        pdf_pattern=r"\.pdf",
        direct_pdf_pattern=r"ajustes/\d{4}/AJ\d+",
        max_depth=2,
        same_domain_only=True,
    ),
    # Atos COTEPE — tabelas de MVA, pauta fiscal, ST específica
    CrawlSource(
        url="https://www.confaz.fazenda.gov.br/legislacao/atos-cotepe",
        folder_name="analista_fiscal",
        description="CONFAZ — Atos COTEPE/ICMS (MVA, pauta fiscal, obrigações acessórias)",
        use_browser=False,
        pdf_pattern=r"\.pdf",
        max_depth=2,
        same_domain_only=True,
    ),

    # =========================================================================
    # 2. ANALISTA CONTÁBIL — CPC e CFC (índices HTML com links para PDF)
    # =========================================================================

    CrawlSource(
        url="https://www.cpc.org.br/CPC/Documentos-Emitidos/Pronunciamentos",
        folder_name="analista_contabil",
        description="CPC — Pronunciamentos Contábeis (CPC 00 ao CPC 48)",
        use_browser=False,
        pdf_pattern=r"\.pdf",
        max_depth=1,
        same_domain_only=False,
    ),
    CrawlSource(
        url="https://www.cpc.org.br/CPC/Documentos-Emitidos/Interpretacoes",
        folder_name="analista_contabil",
        description="CPC — Interpretações Técnicas (ICPC)",
        use_browser=False,
        pdf_pattern=r"\.pdf",
        max_depth=1,
        same_domain_only=False,
    ),
    CrawlSource(
        url="https://www.cpc.org.br/CPC/Documentos-Emitidos/Orientacoes",
        folder_name="analista_contabil",
        description="CPC — Orientações Técnicas (OCPC)",
        use_browser=False,
        pdf_pattern=r"\.pdf",
        max_depth=1,
        same_domain_only=False,
    ),
    CrawlSource(
        url="https://cfc.org.br/tecnica/normas-brasileiras-de-contabilidade/nbc-tg-geral/",
        folder_name="analista_contabil",
        description="CFC — NBC TG Gerais (normas técnicas de contabilidade)",
        use_browser=False,
        pdf_pattern=r"\.pdf",
        max_depth=1,
        same_domain_only=False,
    ),
    CrawlSource(
        url="https://cfc.org.br/tecnica/normas-brasileiras-de-contabilidade/nbc-ta-auditoria-independente/",
        folder_name="analista_contabil",
        description="CFC — NBC TA (Auditoria Independente)",
        use_browser=False,
        pdf_pattern=r"\.pdf",
        max_depth=1,
        same_domain_only=False,
    ),

    # =========================================================================
    # 3. ANALISTA DEPARTAMENTO PESSOAL — eSocial e MTE
    # =========================================================================

    CrawlSource(
        url="https://www.gov.br/esocial/pt-br/documentacao-tecnica/manuais",
        folder_name="analista_departamento_pessoal",
        description="eSocial — Manuais técnicos e layouts de eventos (S-xxxx)",
        use_browser=False,
        pdf_pattern=r"\.pdf",
        max_depth=1,
        same_domain_only=True,
    ),
    CrawlSource(
        url="https://www.gov.br/trabalho-e-emprego/pt-br/acesso-a-informacao/participacao-social/conselhos-e-orgaos-colegiados/ctpp-nrs/portarias-aprovadas-pelo-menos-de-2019",
        folder_name="analista_departamento_pessoal",
        description="MTE — Normas Regulamentadoras (NR-01 a NR-38)",
        use_browser=False,
        pdf_pattern=r"\.pdf",
        max_depth=1,
        same_domain_only=True,
    ),
    CrawlSource(
        url="https://www.gov.br/previdencia/pt-br/assuntos/previdencia-social/contato/legislacao",
        folder_name="analista_departamento_pessoal",
        description="Previdência Social — Legislação previdenciária (decretos e INs)",
        use_browser=False,
        pdf_pattern=r"\.pdf",
        max_depth=1,
        same_domain_only=True,
    ),

    # =========================================================================
    # 4. ANALISTA SOCIETÁRIO — DREI
    # =========================================================================

    CrawlSource(
        url="https://drei.gov.br/pt-br/legislacao/instrucoes-normativas",
        folder_name="analista_societario",
        description="DREI — Instruções Normativas (atos registráveis, tipos societários)",
        use_browser=False,
        pdf_pattern=r"\.pdf",
        max_depth=1,
        same_domain_only=True,
    ),
    CrawlSource(
        url="https://drei.gov.br/pt-br/legislacao/manuais-e-modelos",
        folder_name="analista_societario",
        description="DREI — Manuais e modelos de atos societários",
        use_browser=False,
        pdf_pattern=r"\.pdf",
        max_depth=1,
        same_domain_only=True,
    ),

    # =========================================================================
    # 5. ANALISTA ABERTURA DE EMPRESAS — MA
    # =========================================================================

    CrawlSource(
        url="https://www.jucema.ma.gov.br/legislacao",
        folder_name="analista_abertura_empresas",
        description="JUCEMA — Legislação registral do Maranhão",
        use_browser=False,
        pdf_pattern=r"\.pdf",
        max_depth=1,
        same_domain_only=True,
    ),
    CrawlSource(
        url="https://www.jucema.ma.gov.br/servicos/tabela-de-precos",
        folder_name="analista_abertura_empresas",
        description="JUCEMA — Tabela de preços e documentos exigidos",
        use_browser=False,
        pdf_pattern=r"\.pdf",
        max_depth=1,
        same_domain_only=True,
    ),
    CrawlSource(
        url="https://www.cbm.ma.gov.br/instrucoes-tecnicas/",
        folder_name="analista_abertura_empresas",
        description="CBMMA — Instruções Técnicas (IT-01 a IT-34) — CLCB e AVCB",
        use_browser=False,
        pdf_pattern=r"\.pdf",
        max_depth=1,
        same_domain_only=True,
    ),
    CrawlSource(
        url="https://drei.gov.br/pt-br/legislacao/instrucoes-normativas",
        folder_name="analista_abertura_empresas",
        description="DREI — INs para registro na JUCEMA",
        use_browser=False,
        pdf_pattern=r"\.pdf",
        max_depth=1,
        same_domain_only=True,
    ),
]


# =============================================================================
# Deduplicação
# =============================================================================

def _hash_url(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()

def _hash_content(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def _url_already_downloaded(url: str) -> Optional[dict]:
    resp = (
        _supabase().table("crawl_log")
        .select("id,filename,downloaded_at,drive_file_id")
        .eq("url_hash", _hash_url(url))
        .eq("status", "downloaded")
        .limit(1).execute()
    )
    return resp.data[0] if resp.data else None

def _content_already_exists(content_hash: str) -> Optional[dict]:
    resp = (
        _supabase().table("crawl_log")
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
    if size_kb:       record["file_size_kb"]  = size_kb
    if content_hash:  record["content_hash"]  = content_hash
    if error_msg:     record["error_msg"]     = error_msg
    if status == "downloaded": record["downloaded_at"] = now
    _supabase().table("crawl_log").upsert(record, on_conflict="url_hash").execute()


# =============================================================================
# URL → filename
# =============================================================================

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


# =============================================================================
# Fetch de página
# =============================================================================

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
        logger.error("Playwright não instalado")
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


# =============================================================================
# Extração de links
# =============================================================================

def _extract_pdf_links(html: str, base_url: str, source: CrawlSource) -> list[dict]:
    found, seen = [], set()
    base_domain = urlparse(base_url).netloc

    if source.direct_pdf_pattern:
        for m in re.finditer(source.direct_pdf_pattern, html):
            full = urljoin(base_url, m.group(0))
            if full not in seen:
                seen.add(full)
                found.append({"url": full, "title": ""})

    for m in re.finditer(r'<a\s[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
                         html, re.IGNORECASE | re.DOTALL):
        href, text = m.group(1), re.sub(r"<[^>]+>", "", m.group(2)).strip()
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
        full = urljoin(base_url, m.group(1))
        if urlparse(full).netloc != base_domain: continue
        if full in seen or full == base_url: continue
        if re.search(r"\.(pdf|doc|docx|xls|zip)$", full, re.IGNORECASE): continue
        seen.add(full)
        links.append(full)
    return links[:50]  # aumentado de 20 para 50 (portais JSF têm muitos sublinks)


# =============================================================================
# Download de PDF
# =============================================================================

async def _download_pdf(url: str) -> Optional[bytes]:
    try:
        async with httpx.AsyncClient(
            timeout=60, follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; FinTaxBot/1.0)"}
        ) as c:
            r = await c.get(url)
            r.raise_for_status()
            if not (r.content[:4] == b"%PDF" or "pdf" in r.headers.get("content-type", "")):
                if ".pdf" not in url.lower():
                    return None
            return r.content
    except Exception as e:
        logger.error(f"  ✗ Download {url}: {e}")
        return None


# =============================================================================
# Crawl principal
# =============================================================================

async def crawl_and_upload(source: CrawlSource) -> list[dict]:
    logger.info(f"\n🌐 {source.description or source.url}")
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
            logger.error(f"  ✗ {url}: {e}")
            return
        new = [p for p in _extract_pdf_links(html, url, source)
               if p["url"] not in {x["url"] for x in pdf_links}]
        pdf_links.extend(new)
        if new:
            logger.info(f"  🔗 {len(new)} PDFs em {url}")
        if depth < source.max_depth:
            for sub in _extract_sub_links(html, url):
                await crawl_page(sub, depth + 1)
                await asyncio.sleep(0.5)

    await crawl_page(source.url, 0)
    logger.info(f"  Total: {len(pdf_links)} PDFs encontrados")

    results = []
    for item in pdf_links:
        url, title = item["url"], item.get("title", "")
        filename = _url_to_filename(url, title)

        # C1: URL hash
        existing = _url_already_downloaded(url)
        if existing:
            logger.debug(f"  ⏭ [C1] {filename}")
            _log_crawl(url, filename, source.folder_name, "downloaded",
                       source.url, existing.get("drive_file_id", ""))
            results.append({"file": filename, "status": "skipped", "reason": "url_in_log"})
            continue

        # C2: nome no Drive
        if _pdf_exists_in_folder(svc, filename, folder_id):
            logger.debug(f"  ⏭ [C2] {filename}")
            _log_crawl(url, filename, source.folder_name, "downloaded", source.url)
            results.append({"file": filename, "status": "skipped", "reason": "name_in_drive"})
            continue

        # Download
        logger.info(f"  ↓ {filename}")
        pdf_bytes = await _download_pdf(url)
        if not pdf_bytes:
            _log_crawl(url, filename, source.folder_name, "error", source.url,
                       error_msg="download falhou")
            results.append({"file": filename, "status": "error"})
            continue

        # C3: hash do conteúdo
        c_hash = _hash_content(pdf_bytes)
        dup = _content_already_exists(c_hash)
        if dup:
            logger.debug(f"  ⏭ [C3] conteúdo duplicado de '{dup['filename']}'")
            _log_crawl(url, filename, source.folder_name, "downloaded", source.url,
                       dup.get("drive_file_id", ""), len(pdf_bytes) // 1024, c_hash)
            results.append({"file": filename, "status": "skipped", "reason": "content_duplicate"})
            continue

        # Upload Drive
        try:
            drive_id = _upload_bytes_to_drive(svc, pdf_bytes, filename, folder_id)
            size_kb = len(pdf_bytes) // 1024
            logger.success(f"  ✓ {filename} ({size_kb} KB)")
            _log_crawl(url, filename, source.folder_name, "downloaded",
                       source.url, drive_id, size_kb, c_hash)
            results.append({"file": filename, "status": "uploaded",
                            "drive_id": drive_id, "size_kb": size_kb})
        except Exception as e:
            logger.error(f"  ✗ Upload: {e}")
            _log_crawl(url, filename, source.folder_name, "error",
                       source.url, error_msg=str(e))
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
    logger.info(f"🏁 {up} uploads | {sk} pulados | {err} erros")
    return all_results
