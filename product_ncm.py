"""
product_ncm.py — Mapa produto → NCM para resolução semântica no RAG

Problema resolvido:
  Perguntas como "alíquota de ICMS-ST para refrigerante" não contêm código
  NCM numérico, então _extract_ncms_from_question() retorna [] e a tabela
  kb_ncm_fiscal nunca é consultada. Este módulo resolve nomes de produtos
  para NCMs antes da busca semântica.

Uso:
  from product_ncm import resolve_product_ncms
  ncms = resolve_product_ncms("refrigerante")   # → ["2202"]
"""

import re
from typing import Optional

# ── Mapa canônico produto → lista de NCMs (8 dígitos sem pontos) ─────────────
# Prioridade: NCM mais específico primeiro, NCM de capítulo por último
# Fonte: TIPI (Decreto 11.158/2022) + RICMS-MA (Decreto 19.714/2003)

PRODUCT_NCM_MAP: dict[str, list[str]] = {

    # ── BEBIDAS ────────────────────────────────────────────────────────────────
    "refrigerante":         ["22021000", "22029900", "22029200"],
    "refrigerantes":        ["22021000", "22029900", "22029200"],
    "agua mineral":         ["22011000", "22019000"],
    "agua com gas":         ["22011000"],
    "suco":                 ["20091900", "20099000"],
    "nectar":               ["20099000"],
    "isotônico":            ["22029900"],
    "energético":           ["22029900"],
    "cerveja":              ["22030000"],
    "chope":                ["22030000"],
    "vinho":                ["22042100", "22042900"],
    "cachaça":              ["22087000"],
    "vodka":                ["22089300"],
    "whisky":               ["22083000"],
    "aguardente":           ["22087000"],

    # ── ALIMENTOS ─────────────────────────────────────────────────────────────
    "café":                 ["09011100", "09011200", "21011100"],
    "cafe":                 ["09011100", "09011200", "21011100"],
    "açúcar":               ["17011400", "17019900"],
    "acucar":               ["17011400", "17019900"],
    "farinha de trigo":     ["11010010"],
    "arroz":                ["10061000", "10062000", "10063000", "10064000"],
    "feijão":               ["07133300"],
    "feijao":               ["07133300"],
    "leite":                ["04011000", "04012000", "04013000"],
    "manteiga":             ["04051000"],
    "queijo":               ["04061000", "04062000", "04063000", "04064000"],
    "óleo de soja":         ["15071000", "15079000"],
    "oleo de soja":         ["15071000", "15079000"],
    "óleo de girassol":     ["15121110"],
    "margarina":            ["15171000"],
    "macarrão":             ["19021900"],
    "macarrao":             ["19021900"],
    "biscoito":             ["19053100", "19053200", "19059090"],
    "bolacha":              ["19053100", "19053200"],
    "chocolate":            ["18061000", "18062000", "18063100", "18063200"],
    "sorvete":              ["21050000"],
    "iogurte":              ["04031000"],
    "requeijão":            ["04063000"],
    "presunto":             ["16010000", "02101100"],
    "mortadela":            ["16010000"],
    "salsicha":             ["16013000"],
    "frango":               ["02071100", "02071200", "02071300"],
    "carne bovina":         ["02010000", "02020000"],
    "carne suína":          ["02030000"],
    "peixe":                ["03020000", "03030000"],

    # ── CIGARROS / FUMO ──────────────────────────────────────────────────────
    "cigarro":              ["24022000"],
    "charuto":              ["24021000"],
    "tabaco":               ["24011000", "24012000"],
    "fumo":                 ["24011000", "24012000"],

    # ── COMBUSTÍVEIS ─────────────────────────────────────────────────────────
    "gasolina":             ["27101259", "27101262"],
    "etanol":               ["22071000", "22072000", "27101400"],
    "álcool":               ["22071000", "22072000"],
    "alcool":               ["22071000", "22072000"],
    "diesel":               ["27101921", "27101922"],
    "glp":                  ["27112100"],
    "gás lpg":              ["27112100"],
    "gas lpg":              ["27112100"],
    "querosene":            ["27101911"],
    "lubrificante":         ["27101500", "27101600"],
    "óleo lubrificante":    ["27101500", "27101600"],
    "oleo lubrificante":    ["27101500", "27101600"],

    # ── MEDICAMENTOS ─────────────────────────────────────────────────────────
    "medicamento":          ["30039000", "30049000"],
    "remédio":              ["30039000", "30049000"],
    "remedio":              ["30039000", "30049000"],
    "antibiótico":          ["30041000", "30042000"],
    "antibiotico":          ["30041000", "30042000"],
    "vacina":               ["30021000", "30022000"],
    "soro":                 ["30021000"],
    "anticoncepcionais":    ["30049099"],

    # ── HIGIENE / COSMÉTICOS ─────────────────────────────────────────────────
    "shampoo":              ["33051000"],
    "sabonete":             ["34011100", "34011900"],
    "creme dental":         ["33061000"],
    "pasta de dente":       ["33061000"],
    "desodorante":          ["33079000"],
    "perfume":              ["33030010", "33030020"],
    "maquiagem":            ["33040000"],
    "creme hidratante":     ["33049900"],
    "papel higiênico":      ["48180000"],
    "papel higienico":      ["48180000"],
    "absorvente":           ["96190000"],
    "fraldas":              ["96190000"],

    # ── PRODUTOS DE LIMPEZA ──────────────────────────────────────────────────
    "detergente":           ["34022000", "34029000"],
    "sabão":                ["34012000"],
    "sabao":                ["34012000"],
    "amaciante":            ["38091000"],
    "alvejante":            ["28281000"],
    "desinfetante":         ["38089400"],
    "água sanitária":       ["28281000"],
    "agua sanitaria":       ["28281000"],

    # ── ELETROELETRÔNICOS ────────────────────────────────────────────────────
    "televisão":            ["85287200", "85287900"],
    "televisao":            ["85287200", "85287900"],
    "tv":                   ["85287200", "85287900"],
    "celular":              ["85171200"],
    "smartphone":           ["85171200"],
    "computador":           ["84714100", "84714900"],
    "notebook":             ["84713000"],
    "tablet":               ["84713000"],
    "geladeira":            ["84181000", "84182100"],
    "refrigerador":         ["84181000", "84182100"],
    "fogão":                ["73211100"],
    "fogao":                ["73211100"],
    "microondas":           ["85165000"],
    "lavadora":             ["84501100"],
    "máquina de lavar":     ["84501100"],
    "maquina de lavar":     ["84501100"],
    "ar condicionado":      ["84152000", "84159000"],
    "ventilador":           ["84145900"],

    # ── VEÍCULOS ─────────────────────────────────────────────────────────────
    "automóvel":            ["87032100", "87032210", "87032290"],
    "automovel":            ["87032100", "87032210", "87032290"],
    "carro":                ["87032100", "87032210", "87032290"],
    "motocicleta":          ["87112000", "87113000"],
    "moto":                 ["87112000", "87113000"],
    "caminhão":             ["87042100", "87042200"],
    "caminhao":             ["87042100", "87042200"],
    "ônibus":               ["87022000", "87023000"],
    "onibus":               ["87022000", "87023000"],
    "pneu":                 ["40111000", "40112000", "40119900"],
    "peças para veículos":  ["87089900"],
    "pecas para veiculos":  ["87089900"],

    # ── CONSTRUÇÃO CIVIL ─────────────────────────────────────────────────────
    "cimento":              ["25231000", "25232100", "25232900"],
    "tinta":                ["32089090", "32091000"],
    "vergalhão":            ["72142000"],
    "vergalhao":            ["72142000"],
    "aço":                  ["72142000", "72081000"],
    "aco":                  ["72142000"],
    "telha":                ["68051000", "69050000"],
    "tijolo":               ["69010000"],
    "cal":                  ["25221000"],
    "areia":                ["26200000"],
    "brita":                ["25171000"],

    # ── AGROPECUÁRIA ─────────────────────────────────────────────────────────
    "adubo":                ["31010000", "31021000"],
    "fertilizante":         ["31010000", "31021000"],
    "agrotóxico":           ["38080000"],
    "agrotoxico":           ["38080000"],
    "pesticida":            ["38080000"],
    "herbicida":            ["38082000"],
    "inseticida":           ["38081000"],
    "sementes":             ["12099100", "12099900"],
    "rações":               ["23091000", "23099010"],
    "racoes":               ["23091000", "23099010"],
    "ração":                ["23091000"],
    "racao":                ["23091000"],
    "vacina animal":        ["30023000"],
}

# ── Alias: normaliza entrada do usuário ──────────────────────────────────────

_RE_NORMALIZE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Converte para minúsculas e colapsa espaços."""
    return _RE_NORMALIZE.sub(" ", text.lower().strip())


def resolve_product_ncms(text: str) -> list[str]:
    """
    Tenta resolver nomes de produtos no texto para códigos NCM.

    Estratégia:
      1. Normaliza o texto
      2. Testa cada chave do mapa (frases longas primeiro para evitar
         match prematuro de substring mais curta)
      3. Retorna lista de NCMs únicos, mantendo a ordem de relevância

    Exemplo:
      resolve_product_ncms("alíquota ICMS-ST para refrigerante no MA")
      → ["22021000", "22029900", "22029200"]
    """
    normalized = _normalize(text)
    found: list[str] = []
    seen: set[str] = set()

    # Ordena chaves por comprimento decrescente para preferir match mais específico
    sorted_keys = sorted(PRODUCT_NCM_MAP.keys(), key=len, reverse=True)

    for key in sorted_keys:
        if key in normalized:
            for ncm in PRODUCT_NCM_MAP[key]:
                if ncm not in seen:
                    seen.add(ncm)
                    found.append(ncm)

    return found


def get_ncm_variants(ncm: str) -> list[str]:
    """
    Dado um NCM de 8 dígitos, retorna variações para busca ampla:
      - 8 dígitos (exato)
      - 6 dígitos (posição + subposição)
      - 4 dígitos (posição/capítulo)

    Útil para ampliar a busca quando o NCM exato não retorna resultado.
    """
    digits = re.sub(r"\D", "", ncm)
    variants = []
    if len(digits) >= 8:
        variants.append(digits[:8])
    if len(digits) >= 6:
        variants.append(digits[:6])
    if len(digits) >= 4:
        variants.append(digits[:4])
    return list(dict.fromkeys(variants))  # dedup mantendo ordem
