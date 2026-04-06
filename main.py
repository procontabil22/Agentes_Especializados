"""
main.py — FastAPI FinTax Agents
O /health responde imediatamente sem depender de nenhum serviço externo.
Todos os clientes externos são inicializados lazy, apenas quando usados.
"""
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI, HTTPException, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

from settings import settings  # ← flat import

# ── Estado global ─────────────────────────────────────────────────────────────
_index_status = {"last_run": None, "running": False, "last_report": None}
_start_time = datetime.utcnow().isoformat()

scheduler = AsyncIOScheduler()


async def _run_indexing_job(folder: str | None = None):
    if _index_status["running"]:
        logger.warning("Indexação já em andamento")
        return
    _index_status["running"] = True
    _index_status["last_run"] = datetime.utcnow().isoformat()
    try:
        from orchestrator import run_indexing  # ← flat import
        report = await run_indexing(folder_filter=folder)
        _index_status["last_report"] = report
    finally:
        _index_status["running"] = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        parts = settings.AUTO_DOWNLOAD_CRON.split()
        if len(parts) == 5:
            scheduler.add_job(
                _run_indexing_job,
                CronTrigger(minute=parts[0], hour=parts[1], day=parts[2],
                            month=parts[3], day_of_week=parts[4]),
                id="auto_index", replace_existing=True,
            )
            scheduler.start()
            logger.info(f"Scheduler iniciado. Cron: {settings.AUTO_DOWNLOAD_CRON}")
        else:
            logger.warning(f"AUTO_DOWNLOAD_CRON inválido: '{settings.AUTO_DOWNLOAD_CRON}' — scheduler desabilitado")
    except Exception as e:
        logger.warning(f"Scheduler não iniciado: {e}")
    yield
    try:
        scheduler.shutdown()
    except Exception:
        pass


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="FinTax Agents API",
    description="Microserviço de agentes IA especializados — FinTax",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _check_auth(x_api_key: str | None):
    if x_api_key != settings.API_SECRET_KEY:
        raise HTTPException(status_code=401, detail="API key inválida")


# ── Schemas ───────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    history: list[dict] = []

class IndexRequest(BaseModel):
    folder: Optional[str] = None
    download_sources: bool = True


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "fintax-agents",
        "version": "1.0.0",
        "started_at": _start_time,
    }


@app.get("/agents")
def agents_list():
    from agents import list_agents  # ← flat import
    return {"agents": list_agents()}


@app.get("/agents/{agent_id}")
def agent_detail(agent_id: str):
    from agents import get_agent  # ← flat import
    agent = get_agent(agent_id)
    if not agent:
        raise HTTPException(404, f"Agente '{agent_id}' não encontrado")
    return {
        "id": agent.id, "name": agent.name, "icon": agent.icon,
        "description": agent.description, "table_name": agent.table_name,
        "color": agent.color,
    }


@app.post("/chat/{agent_id}")
async def chat(
    agent_id: str,
    body: ChatRequest,
    x_api_key: Optional[str] = Header(None),
):
    _check_auth(x_api_key)
    from agents import get_agent  # ← flat import
    if not get_agent(agent_id):
        raise HTTPException(404, f"Agente '{agent_id}' não encontrado")
    from chat import ask_agent  # ← flat import
    return await ask_agent(agent_id, body.question, body.history)


@app.post("/index")
async def trigger_index(
    body: IndexRequest,
    background_tasks: BackgroundTasks,
    x_api_key: Optional[str] = Header(None),
):
    _check_auth(x_api_key)
    if _index_status["running"]:
        raise HTTPException(409, "Indexação já em andamento")
    background_tasks.add_task(_run_indexing_job, body.folder)
    return {"status": "started", "folder": body.folder or "todas",
            "message": "Indexação iniciada em background. Consulte /index/status."}


@app.post("/index/{folder_name}")
async def trigger_index_folder(
    folder_name: str,
    background_tasks: BackgroundTasks,
    x_api_key: Optional[str] = Header(None),
):
    _check_auth(x_api_key)
    background_tasks.add_task(_run_indexing_job, folder_name)
    return {"status": "started", "folder": folder_name}


@app.get("/index/status")
def index_status(x_api_key: Optional[str] = Header(None)):
    _check_auth(x_api_key)
    return _index_status


@app.get("/crawler/sources")
def crawler_sources(x_api_key: Optional[str] = Header(None)):
    _check_auth(x_api_key)
    from crawler import CRAWL_SOURCES  # ← flat import
    return {"sources": [
        {"url": s.url, "folder": s.folder_name,
         "description": s.description, "use_browser": s.use_browser}
        for s in CRAWL_SOURCES
    ]}


@app.post("/crawler/run")
async def crawler_run(
    background_tasks: BackgroundTasks,
    folder: Optional[str] = None,
    x_api_key: Optional[str] = Header(None),
):
    _check_auth(x_api_key)

    async def _job():
        from crawler import run_crawler  # ← flat import
        results = await run_crawler(source_filter=folder)
        uploaded = sum(1 for r in results if r["status"] == "uploaded")
        if uploaded > 0:
            await _run_indexing_job(folder)

    background_tasks.add_task(_job)
    return {"status": "started", "folder": folder or "todas"}


@app.get("/crawler/log")
def crawler_log(
    folder: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    x_api_key: Optional[str] = Header(None),
):
    _check_auth(x_api_key)
    from supabase import create_client
    sb = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)
    query = (
        sb.table("crawl_log")
        .select("url,filename,folder_name,status,file_size_kb,downloaded_at,last_checked_at,error_msg")
        .order("last_checked_at", desc=True)
        .limit(limit)
    )
    if folder: query = query.eq("folder_name", folder)
    if status: query = query.eq("status", status)
    data = query.execute().data or []
    return {
        "summary": {
            "total": len(data),
            "downloaded": sum(1 for r in data if r["status"] == "downloaded"),
            "skipped":    sum(1 for r in data if r["status"] == "skipped"),
            "error":      sum(1 for r in data if r["status"] == "error"),
        },
        "records": data,
    }
