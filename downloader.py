"""
app/downloader.py
Download automático de fontes públicas para o Google Drive.
Cada agente tem sua lista de fontes com URL, nome e pasta de destino.
"""
import io, time
from dataclasses import dataclass

import httpx
from loguru import logger

from app.gdrive import _get_service, _get_or_create_folder, _pdf_exists_in_folder, _upload_bytes_to_drive
from config.settings import settings


@dataclass
class Source:
    url: str
    filename: str
    folder_name: str
    description: str = ""


# ════════════════════════════════════════════════════════════════════════════
# FONTES PÚBLICAS POR AGENTE
# Adicione URLs aqui — o microserviço baixará automaticamente no cron semanal
# ════════════════════════════════════════════════════════════════════════════
SOURCES: list[Source] = [

    # ── 1. CONTÁBIL ──────────────────────────────────────────────────────────
    Source(
        url="https://www.planalto.gov.br/ccivil_03/leis/l6404consol.htm",
        filename="Lei_6404_1976_SA_Compilada.pdf",
        folder_name="contabil",
        description="Lei 6.404/1976 — Lei das S.A. (compilada)",
    ),
    Source(
        url="https://cfc.org.br/tecnica/normas-brasileiras-de-contabilidade/nbc-tg-geral/",
        filename="NBC_TG_Indice_CFC.pdf",
        folder_name="contabil",
        description="CFC — Índice NBC TG (normas técnicas gerais)",
    ),

    # ── 2. FISCAL ────────────────────────────────────────────────────────────
    Source(
        url="https://www.planalto.gov.br/ccivil_03/leis/lcp/lcp214.htm",
        filename="LC_214_2024_Reforma_Tributaria.pdf",
        folder_name="fiscal",
        description="LC 214/2024 — CBS, IBS, Imposto Seletivo",
    ),
    Source(
        url="https://www.planalto.gov.br/ccivil_03/leis/lcp/lcp087.htm",
        filename="LC_087_1996_Lei_Kandir_ICMS.pdf",
        folder_name="fiscal",
        description="LC 87/1996 — Lei Kandir (ICMS)",
    ),
    Source(
        url="https://www.planalto.gov.br/ccivil_03/leis/lcp/lcp123.htm",
        filename="LC_123_2006_Simples_Nacional_Compilada.pdf",
        folder_name="fiscal",
        description="LC 123/2006 — Simples Nacional (compilada)",
    ),
    Source(
        url="https://www.planalto.gov.br/ccivil_03/leis/2002/l10637.htm",
        filename="Lei_10637_2002_PIS_Nao_Cumulativo.pdf",
        folder_name="fiscal",
        description="Lei 10.637/2002 — PIS não-cumulativo (LR)",
    ),
    Source(
        url="https://www.planalto.gov.br/ccivil_03/leis/2003/l10833.htm",
        filename="Lei_10833_2003_COFINS_Nao_Cumulativa.pdf",
        folder_name="fiscal",
        description="Lei 10.833/2003 — COFINS não-cumulativa (LR)",
    ),
    Source(
        url="https://www.planalto.gov.br/ccivil_03/leis/l9718compilada.htm",
        filename="Lei_9718_1998_PIS_COFINS_Cumulativo.pdf",
        folder_name="fiscal",
        description="Lei 9.718/1998 — PIS/COFINS cumulativo (LP)",
    ),

    # ── 3. PESSOAL ───────────────────────────────────────────────────────────
    Source(
        url="https://www.planalto.gov.br/ccivil_03/decreto-lei/del5452.htm",
        filename="CLT_Decreto_Lei_5452_1943_Compilada.pdf",
        folder_name="pessoal",
        description="CLT — Consolidação das Leis do Trabalho (compilada)",
    ),
    Source(
        url="https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2017/lei/l13467.htm",
        filename="Lei_13467_2017_Reforma_Trabalhista.pdf",
        folder_name="pessoal",
        description="Lei 13.467/2017 — Reforma Trabalhista",
    ),
    Source(
        url="https://www.planalto.gov.br/ccivil_03/constituicao/constituicao.htm",
        filename="CF_1988_Direitos_Sociais_Art_7.pdf",
        folder_name="pessoal",
        description="CF/1988 — Art. 7° (direitos trabalhistas constitucionais)",
    ),

    # ── 4. SOCIETÁRIO ────────────────────────────────────────────────────────
    Source(
        url="https://www.planalto.gov.br/ccivil_03/leis/2002/l10406compilada.htm",
        filename="Codigo_Civil_2002_Compilado.pdf",
        folder_name="societario",
        description="Código Civil 2002 — Compilado (inclui direito societário)",
    ),
    Source(
        url="https://www.planalto.gov.br/ccivil_03/leis/l6404consol.htm",
        filename="Lei_6404_1976_SA_Societario.pdf",
        folder_name="societario",
        description="Lei 6.404/1976 — Lei das S.A. (visão societária)",
    ),
    Source(
        url="https://www.planalto.gov.br/ccivil_03/_ato2019-2022/2021/lei/l14195.htm",
        filename="Lei_14195_2021_SLU_EIRELI_Desburocratizacao.pdf",
        folder_name="societario",
        description="Lei 14.195/2021 — SLU, extinção da EIRELI, desburocratização",
    ),
    Source(
        url="https://www.planalto.gov.br/ccivil_03/leis/l5764.htm",
        filename="Lei_5764_1971_Cooperativas.pdf",
        folder_name="societario",
        description="Lei 5.764/1971 — Cooperativas",
    ),
    Source(
        url="https://www.planalto.gov.br/ccivil_03/leis/l8934.htm",
        filename="Lei_8934_1994_Registro_Mercantil.pdf",
        folder_name="societario",
        description="Lei 8.934/1994 — Registro Mercantil",
    ),
    Source(
        url="https://www.planalto.gov.br/ccivil_03/_ato2011-2014/2013/lei/l12846.htm",
        filename="Lei_12846_2013_Lei_Anticorrupcao.pdf",
        folder_name="societario",
        description="Lei 12.846/2013 — Lei Anticorrupção",
    ),

    # ── 5. ABERTURA EMPRESAS — MARANHÃO ──────────────────────────────────────
    Source(
        url="https://www.planalto.gov.br/ccivil_03/leis/l8934.htm",
        filename="Lei_8934_1994_Registro_Mercantil_JUCEMA.pdf",
        folder_name="abertura_ma",
        description="Lei 8.934/1994 — Registro Mercantil (JUCEMA)",
    ),
    Source(
        url="https://www.planalto.gov.br/ccivil_03/leis/lcp/lcp123.htm",
        filename="LC_123_2006_Simples_Nacional_Abertura.pdf",
        folder_name="abertura_ma",
        description="LC 123/2006 — Simples Nacional (opção na abertura)",
    ),
    Source(
        url="https://www.planalto.gov.br/ccivil_03/_ato2019-2022/2021/lei/l14195.htm",
        filename="Lei_14195_2021_REDESIM_Abertura.pdf",
        folder_name="abertura_ma",
        description="Lei 14.195/2021 — REDESIM, desburocratização da abertura",
    ),
]


def download_public_sources() -> list[dict]:
    """
    Percorre SOURCES, baixa cada PDF via HTTP e faz upload para o Drive.
    Pula arquivos que já existem (mesmo nome na pasta correta do Drive).
    """
    if not settings.AUTO_DOWNLOAD_ENABLED:
        logger.info("Download automático desabilitado (AUTO_DOWNLOAD_ENABLED=false)")
        return []

    svc = _get_service()
    results = []

    # Mapeia pasta → folder_id (cria se não existir)
    folder_ids: dict[str, str] = {}
    for source in SOURCES:
        if source.folder_name not in folder_ids:
            fid = _get_or_create_folder(svc, source.folder_name, settings.GDRIVE_ROOT_FOLDER_ID)
            folder_ids[source.folder_name] = fid

    for source in SOURCES:
        folder_id = folder_ids[source.folder_name]

        if _pdf_exists_in_folder(svc, source.filename, folder_id):
            logger.debug(f"  ⏭ {source.filename} já existe")
            results.append({"file": source.filename, "status": "skipped"})
            continue

        logger.info(f"  ↓ Baixando: {source.description}")
        try:
            with httpx.Client(timeout=60, follow_redirects=True) as client:
                resp = client.get(source.url)
                resp.raise_for_status()
                pdf_bytes = resp.content

            drive_id = _upload_bytes_to_drive(svc, pdf_bytes, source.filename, folder_id)
            logger.success(f"  ✓ {source.filename} → Drive {drive_id}")
            results.append({"file": source.filename, "status": "uploaded", "drive_id": drive_id})
        except Exception as e:
            logger.error(f"  ✗ {source.filename}: {e}")
            results.append({"file": source.filename, "status": "error", "error": str(e)})

        time.sleep(1)

    return results
