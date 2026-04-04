"""
downloader.py
Download automático de fontes públicas para o Google Drive.
Fontes: legislação federal (Planalto), CFC, RFB, eSocial, DREI.
Estas são fontes HTTP diretas (sem necessidade de crawler/Playwright).
Portais JSF e com navegação dinâmica estão no crawler.py.
"""
import time
from dataclasses import dataclass

import httpx
from loguru import logger

from gdrive import _get_service, _get_or_create_folder, _pdf_exists_in_folder, _upload_bytes_to_drive  # flat import
from settings import settings  # flat import


@dataclass
class Source:
    url: str
    filename: str
    folder_name: str
    description: str = ""


# =============================================================================
# FONTES PÚBLICAS POR AGENTE
# Apenas fontes acessíveis via HTTP direto (sem JavaScript / login).
# Portais com JSF ou navegação dinâmica → use crawler.py
# =============================================================================
SOURCES: list[Source] = [

    # =========================================================================
    # 1. ANALISTA CONTÁBIL
    # =========================================================================

    Source(
        url="https://www.planalto.gov.br/ccivil_03/leis/l6404consol.htm",
        filename="Lei_6404_1976_SA_Compilada.pdf",
        folder_name="contabil",
        description="Lei 6.404/1976 — Lei das S.A. compilada",
    ),
    Source(
        url="https://www.planalto.gov.br/ccivil_03/_ato2007-2010/2007/lei/l11638.htm",
        filename="Lei_11638_2007_Convergencia_IFRS.pdf",
        folder_name="contabil",
        description="Lei 11.638/2007 — Convergência às normas IFRS",
    ),
    Source(
        url="https://www.planalto.gov.br/ccivil_03/_ato2007-2010/2009/lei/l11941.htm",
        filename="Lei_11941_2009_RTT_Transicao_Tributaria.pdf",
        folder_name="contabil",
        description="Lei 11.941/2009 — RTT (Regime Tributário de Transição)",
    ),
    Source(
        url="https://www.planalto.gov.br/ccivil_03/decreto-lei/del1598.htm",
        filename="DL_1598_1977_IRPJ_Lucro_Real.pdf",
        folder_name="contabil",
        description="Decreto-Lei 1.598/1977 — IRPJ base para LALUR/LACS",
    ),
    # CFC — NBC TG (normas técnicas de contabilidade)
    Source(
        url="https://cfc.org.br/tecnica/normas-brasileiras-de-contabilidade/nbc-tg-geral/",
        filename="NBC_TG_Indice_CFC.pdf",
        folder_name="contabil",
        description="CFC — Índice NBC TG (normas técnicas gerais)",
    ),
    Source(
        url="https://cfc.org.br/tecnica/normas-brasileiras-de-contabilidade/nbc-tg-estrutura-conceitual/",
        filename="NBC_TG_Estrutura_Conceitual_CFC.pdf",
        folder_name="contabil",
        description="CFC — NBC TG Estrutura Conceitual (CPC 00 R2)",
    ),
    # Resolução CFC ITG 1000 — PMEs
    Source(
        url="https://cfc.org.br/tecnica/normas-brasileiras-de-contabilidade/nbc-tg-1000-contabilidade-para-pequenas-e-medias-empresas/",
        filename="NBC_TG_1000_PME_CFC.pdf",
        folder_name="contabil",
        description="CFC — NBC TG 1000 (PMEs — pequenas e médias empresas)",
    ),

    # =========================================================================
    # 2. ANALISTA FISCAL
    # =========================================================================

    # Reforma Tributária
    Source(
        url="https://www.planalto.gov.br/ccivil_03/leis/lcp/lcp214.htm",
        filename="LC_214_2024_Reforma_Tributaria_CBS_IBS_IS.pdf",
        folder_name="fiscal",
        description="LC 214/2024 — CBS, IBS, Imposto Seletivo (Reforma Tributária)",
    ),
    Source(
        url="https://www.planalto.gov.br/ccivil_03/constituicao/emendas/emc/emc132.htm",
        filename="EC_132_2023_Reforma_Tributaria_Emenda.pdf",
        folder_name="fiscal",
        description="EC 132/2023 — Emenda Constitucional da Reforma Tributária",
    ),
    # ICMS federal
    Source(
        url="https://www.planalto.gov.br/ccivil_03/leis/lcp/lcp087.htm",
        filename="LC_087_1996_Lei_Kandir_ICMS.pdf",
        folder_name="fiscal",
        description="LC 87/1996 — Lei Kandir (ICMS)",
    ),
    # Simples Nacional
    Source(
        url="https://www.planalto.gov.br/ccivil_03/leis/lcp/lcp123.htm",
        filename="LC_123_2006_Simples_Nacional_Compilada.pdf",
        folder_name="fiscal",
        description="LC 123/2006 — Simples Nacional compilada",
    ),
    # PIS / COFINS
    Source(
        url="https://www.planalto.gov.br/ccivil_03/leis/2002/l10637.htm",
        filename="Lei_10637_2002_PIS_Nao_Cumulativo.pdf",
        folder_name="fiscal",
        description="Lei 10.637/2002 — PIS não-cumulativo (Lucro Real)",
    ),
    Source(
        url="https://www.planalto.gov.br/ccivil_03/leis/2003/l10833.htm",
        filename="Lei_10833_2003_COFINS_Nao_Cumulativa.pdf",
        folder_name="fiscal",
        description="Lei 10.833/2003 — COFINS não-cumulativa (Lucro Real)",
    ),
    Source(
        url="https://www.planalto.gov.br/ccivil_03/leis/l9718compilada.htm",
        filename="Lei_9718_1998_PIS_COFINS_Cumulativo.pdf",
        folder_name="fiscal",
        description="Lei 9.718/1998 — PIS/COFINS cumulativo (Lucro Presumido)",
    ),
    # IRPJ / CSLL
    Source(
        url="https://www.planalto.gov.br/ccivil_03/decreto/d3000compilado.htm",
        filename="RIR_Decreto_3000_1999_IRPJ_CSLL.pdf",
        folder_name="fiscal",
        description="RIR — Regulamento do IRPJ (Decreto 3.000/1999 compilado)",
    ),
    Source(
        url="https://www.planalto.gov.br/ccivil_03/leis/l7689.htm",
        filename="Lei_7689_1988_CSLL.pdf",
        folder_name="fiscal",
        description="Lei 7.689/1988 — CSLL (Contribuição Social sobre Lucro Líquido)",
    ),
    # IPI
    Source(
        url="https://www.planalto.gov.br/ccivil_03/_ato2007-2010/2010/decreto/d7212.htm",
        filename="RIPI_Decreto_7212_2010_IPI.pdf",
        folder_name="fiscal",
        description="RIPI — Regulamento do IPI (Decreto 7.212/2010)",
    ),
    # ISS
    Source(
        url="https://www.planalto.gov.br/ccivil_03/leis/lcp/lcp116.htm",
        filename="LC_116_2003_ISS.pdf",
        folder_name="fiscal",
        description="LC 116/2003 — ISS (Imposto sobre Serviços)",
    ),
    # IOF
    Source(
        url="https://www.planalto.gov.br/ccivil_03/decreto/d6306.htm",
        filename="Decreto_6306_2007_IOF.pdf",
        folder_name="fiscal",
        description="Decreto 6.306/2007 — Regulamento do IOF",
    ),
    # CTN
    Source(
        url="https://www.planalto.gov.br/ccivil_03/leis/l5172compilado.htm",
        filename="CTN_Lei_5172_1966_Codigo_Tributario_Nacional.pdf",
        folder_name="fiscal",
        description="CTN — Código Tributário Nacional compilado",
    ),

    # =========================================================================
    # 3. ANALISTA DEPARTAMENTO PESSOAL
    # =========================================================================

    Source(
        url="https://www.planalto.gov.br/ccivil_03/decreto-lei/del5452.htm",
        filename="CLT_Decreto_Lei_5452_1943_Compilada.pdf",
        folder_name="pessoal",
        description="CLT — Consolidação das Leis do Trabalho compilada",
    ),
    Source(
        url="https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2017/lei/l13467.htm",
        filename="Lei_13467_2017_Reforma_Trabalhista.pdf",
        folder_name="pessoal",
        description="Lei 13.467/2017 — Reforma Trabalhista",
    ),
    Source(
        url="https://www.planalto.gov.br/ccivil_03/_ato2019-2022/2022/lei/l14442.htm",
        filename="Lei_14442_2022_Teletrabalho_Alimentacao.pdf",
        folder_name="pessoal",
        description="Lei 14.442/2022 — Teletrabalho e vale-alimentação",
    ),
    Source(
        url="https://www.planalto.gov.br/ccivil_03/leis/l8036consol.htm",
        filename="Lei_8036_1990_FGTS_Compilada.pdf",
        folder_name="pessoal",
        description="Lei 8.036/1990 — FGTS compilada",
    ),
    Source(
        url="https://www.planalto.gov.br/ccivil_03/leis/l8212cons.htm",
        filename="Lei_8212_1991_INSS_Custeio_Previdencia.pdf",
        folder_name="pessoal",
        description="Lei 8.212/1991 — Custeio da Previdência Social (INSS patronal)",
    ),
    Source(
        url="https://www.planalto.gov.br/ccivil_03/leis/l8213cons.htm",
        filename="Lei_8213_1991_Beneficios_Previdenciarios.pdf",
        folder_name="pessoal",
        description="Lei 8.213/1991 — Benefícios previdenciários (auxílio, aposentadoria)",
    ),
    Source(
        url="https://www.planalto.gov.br/ccivil_03/leis/l7998.htm",
        filename="Lei_7998_1990_Seguro_Desemprego_FAT.pdf",
        folder_name="pessoal",
        description="Lei 7.998/1990 — Seguro-desemprego e FAT",
    ),
    Source(
        url="https://www.planalto.gov.br/ccivil_03/leis/l4923.htm",
        filename="Lei_4923_1965_Aviso_Previo.pdf",
        folder_name="pessoal",
        description="Lei 4.923/1965 — Aviso prévio proporcional (complementada pela Lei 12.506/2011)",
    ),
    Source(
        url="https://www.planalto.gov.br/ccivil_03/_ato2011-2014/2011/lei/l12506.htm",
        filename="Lei_12506_2011_Aviso_Previo_Proporcional.pdf",
        folder_name="pessoal",
        description="Lei 12.506/2011 — Aviso prévio proporcional ao tempo de serviço",
    ),
    Source(
        url="https://www.planalto.gov.br/ccivil_03/constituicao/constituicao.htm",
        filename="CF_1988_Completa.pdf",
        folder_name="pessoal",
        description="CF/1988 — Constituição Federal completa (Art. 7° — direitos trabalhistas)",
    ),
    # Decreto que regulamenta o eSocial
    Source(
        url="https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2015/decreto/d8373.htm",
        filename="Decreto_8373_2014_eSocial.pdf",
        folder_name="pessoal",
        description="Decreto 8.373/2014 — Institui o eSocial",
    ),
    # Desoneração da folha
    Source(
        url="https://www.planalto.gov.br/ccivil_03/_ato2011-2014/2011/lei/l12546.htm",
        filename="Lei_12546_2011_Desoneracao_Folha.pdf",
        folder_name="pessoal",
        description="Lei 12.546/2011 — Desoneração da folha de pagamento",
    ),

    # =========================================================================
    # 4. ANALISTA SOCIETÁRIO
    # =========================================================================

    Source(
        url="https://www.planalto.gov.br/ccivil_03/leis/2002/l10406compilada.htm",
        filename="Codigo_Civil_2002_Compilado.pdf",
        folder_name="societario",
        description="Código Civil 2002 compilado — sociedades simples e limitadas",
    ),
    Source(
        url="https://www.planalto.gov.br/ccivil_03/leis/l6404consol.htm",
        filename="Lei_6404_1976_SA_Compilada_Societario.pdf",
        folder_name="societario",
        description="Lei 6.404/1976 — Lei das S.A. compilada",
    ),
    Source(
        url="https://www.planalto.gov.br/ccivil_03/_ato2019-2022/2021/lei/l14195.htm",
        filename="Lei_14195_2021_SLU_REDESIM_Desburocratizacao.pdf",
        folder_name="societario",
        description="Lei 14.195/2021 — SLU, extinção EIRELI, REDESIM",
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
        description="Lei 8.934/1994 — Registro Público Mercantil",
    ),
    Source(
        url="https://www.planalto.gov.br/ccivil_03/_ato2011-2014/2013/lei/l12846.htm",
        filename="Lei_12846_2013_Lei_Anticorrupcao.pdf",
        folder_name="societario",
        description="Lei 12.846/2013 — Lei Anticorrupção",
    ),
    Source(
        url="https://www.planalto.gov.br/ccivil_03/leis/l6385.htm",
        filename="Lei_6385_1976_CVM_Mercado_Capitais.pdf",
        folder_name="societario",
        description="Lei 6.385/1976 — CVM e mercado de capitais",
    ),
    Source(
        url="https://www.planalto.gov.br/ccivil_03/_ato2019-2022/2019/lei/l13874.htm",
        filename="Lei_13874_2019_Liberdade_Economica.pdf",
        folder_name="societario",
        description="Lei 13.874/2019 — Declaração de Direitos de Liberdade Econômica",
    ),
    Source(
        url="https://www.planalto.gov.br/ccivil_03/leis/lcp/lcp128.htm",
        filename="LC_128_2008_MEI_Microempreendedor.pdf",
        folder_name="societario",
        description="LC 128/2008 — MEI (Microempreendedor Individual)",
    ),
    # Lei de Falências
    Source(
        url="https://www.planalto.gov.br/ccivil_03/_ato2004-2006/2005/lei/l11101.htm",
        filename="Lei_11101_2005_Falencias_Recuperacao_Judicial.pdf",
        folder_name="societario",
        description="Lei 11.101/2005 — Falências e Recuperação Judicial",
    ),

    # =========================================================================
    # 5. ANALISTA ABERTURA DE EMPRESAS — MARANHÃO
    # =========================================================================

    Source(
        url="https://www.planalto.gov.br/ccivil_03/leis/l8934.htm",
        filename="Lei_8934_1994_Registro_Mercantil_JUCEMA.pdf",
        folder_name="abertura_ma",
        description="Lei 8.934/1994 — Registro Mercantil (base JUCEMA)",
    ),
    Source(
        url="https://www.planalto.gov.br/ccivil_03/leis/lcp/lcp123.htm",
        filename="LC_123_2006_Simples_Nacional_Abertura.pdf",
        folder_name="abertura_ma",
        description="LC 123/2006 — Simples Nacional (opção na abertura)",
    ),
    Source(
        url="https://www.planalto.gov.br/ccivil_03/_ato2019-2022/2021/lei/l14195.htm",
        filename="Lei_14195_2021_REDESIM_Abertura_MA.pdf",
        folder_name="abertura_ma",
        description="Lei 14.195/2021 — REDESIM e desburocratização da abertura",
    ),
    Source(
        url="https://www.planalto.gov.br/ccivil_03/_ato2019-2022/2019/lei/l13874.htm",
        filename="Lei_13874_2019_Liberdade_Economica_Abertura.pdf",
        folder_name="abertura_ma",
        description="Lei 13.874/2019 — Liberdade Econômica (licenças e alvarás)",
    ),
    Source(
        url="https://www.planalto.gov.br/ccivil_03/leis/lcp/lcp128.htm",
        filename="LC_128_2008_MEI_Abertura_MA.pdf",
        folder_name="abertura_ma",
        description="LC 128/2008 — MEI (Microempreendedor Individual)",
    ),
    Source(
        url="https://www.planalto.gov.br/ccivil_03/leis/2002/l10406compilada.htm",
        filename="Codigo_Civil_2002_Tipos_Societarios_Abertura.pdf",
        folder_name="abertura_ma",
        description="Código Civil 2002 — tipos societários (LTDA, SLU, Simples)",
    ),
    Source(
        url="https://www.planalto.gov.br/ccivil_03/leis/lcp/lcp116.htm",
        filename="LC_116_2003_ISS_Abertura_MA.pdf",
        folder_name="abertura_ma",
        description="LC 116/2003 — ISS (incidência na abertura de prestadores de serviço)",
    ),
    # Lei de vigilância sanitária federal
    Source(
        url="https://www.planalto.gov.br/ccivil_03/leis/l9782.htm",
        filename="Lei_9782_1999_ANVISA_Vigilancia_Sanitaria.pdf",
        folder_name="abertura_ma",
        description="Lei 9.782/1999 — ANVISA (vigilância sanitária federal)",
    ),
]


def download_public_sources() -> list[dict]:
    """
    Percorre SOURCES, baixa cada documento via HTTP e faz upload para o Drive.
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

        logger.info(f"  ↓ {source.description}")
        try:
            with httpx.Client(timeout=60, follow_redirects=True,
                              headers={"User-Agent": "Mozilla/5.0 (compatible; FinTaxBot/1.0)"}) as client:
                resp = client.get(source.url)
                resp.raise_for_status()
                content = resp.content

            drive_id = _upload_bytes_to_drive(svc, content, source.filename, folder_id)
            logger.success(f"  ✓ {source.filename} → Drive {drive_id}")
            results.append({"file": source.filename, "status": "uploaded", "drive_id": drive_id})

        except Exception as e:
            logger.error(f"  ✗ {source.filename}: {e}")
            results.append({"file": source.filename, "status": "error", "error": str(e)})

        time.sleep(1)

    return results
