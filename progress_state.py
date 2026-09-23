"""
progress_state.py — Estado compartilhado do progresso da indexação em andamento.

Lido pelo endpoint GET /index/progress (main.py), atualizado por
orchestrator.py e pipeline.py durante o processamento de cada arquivo.

Módulo global simples: a indexação roda numa thread separada
(asyncio.to_thread, ver main.py) enquanto o event loop principal continua
livre pra responder requisições -- inclusive as de progresso. Um dict global
com atribuições atômicas de chave é visível e seguro o bastante entre threads
nesse cenário (protegido pelo GIL do CPython; não há leitura+escrita
composta que precise de lock).
"""
from datetime import datetime

# (chave interna, rótulo em pt-BR pra exibir) -- ordem = ordem real do pipeline
# (ver pipeline.py: process_pdf = Fase 1, index_from_json = Fase 2).
ETAPAS = [
    ("baixando", "Baixando do Google Drive"),
    ("convertendo", "Convertendo documento (Docling)"),
    ("extraindo_json", "Extraindo dados estruturados (IA)"),
    ("salvando_json", "Salvando JSON no Drive"),
    ("gerando_embeddings", "Gerando embeddings e gravando na tabela vetorizada"),
]
_ETAPA_LABEL = dict(ETAPAS)
_ETAPA_INDICE = {chave: i for i, (chave, _) in enumerate(ETAPAS)}

_progress: dict = {
    "running": False,
    "folder": None,
    "file_name": None,
    "file_index": 0,
    "file_total": 0,
    "stage": None,
    "stage_label": None,
    "stage_index": None,
    "stage_total": len(ETAPAS),
    "started_at": None,
    "updated_at": None,
}


def _tocar():
    _progress["updated_at"] = datetime.utcnow().isoformat()


def iniciar(folder: str | None):
    _progress.update({
        "running": True,
        "folder": folder or "todas",
        "file_name": None,
        "file_index": 0,
        "file_total": 0,
        "stage": None,
        "stage_label": None,
        "stage_index": None,
        "started_at": datetime.utcnow().isoformat(),
    })
    _tocar()


def pasta(folder_name: str, total: int):
    """Chamado ao entrar em cada pasta (folder_filter sempre traz só uma; 'todas' passa por várias)."""
    _progress.update({"folder": folder_name, "file_total": total, "file_index": 0})
    _tocar()


def proximo_arquivo(file_name: str, index: int):
    _progress.update({"file_name": file_name, "file_index": index, "stage": None, "stage_label": None, "stage_index": None})
    _tocar()


def etapa(stage: str):
    _progress.update({
        "stage": stage,
        "stage_label": _ETAPA_LABEL.get(stage, stage),
        "stage_index": _ETAPA_INDICE.get(stage),
    })
    _tocar()


def finalizar():
    _progress.update({"running": False, "file_name": None, "stage": None, "stage_label": None, "stage_index": None})
    _tocar()


def get() -> dict:
    return dict(_progress)
