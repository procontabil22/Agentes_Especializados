"""
settings.py v2.0 — Centraliza variáveis de ambiente
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):

    # ── OpenAI (embeddings + LLM fallback) ───────────────────────────────────
    OPENAI_API_KEY:       Optional[str] = None
    EMBEDDING_MODEL:      str = "text-embedding-3-small"
    EMBEDDING_DIMENSIONS: int = 1536

    # ── LLMs (prioridade: OpenAI → Anthropic → Gemini → Grok → DeepSeek) ────
    ANTHROPIC_API_KEY: Optional[str] = None
    GEMINI_API_KEY:    Optional[str] = None
    GROK_API_KEY:      Optional[str] = None
    DEEPSEEK_API_KEY:  Optional[str] = None

    # ── Supabase ──────────────────────────────────────────────────────────────
    SUPABASE_URL:         str = ""
    SUPABASE_SERVICE_KEY: str = ""

    # ── Google Drive ──────────────────────────────────────────────────────────
    GOOGLE_CREDENTIALS_JSON: str = ""
    GDRIVE_ROOT_FOLDER_ID:   str = ""

    # ── Mapeamento pasta → tabela ─────────────────────────────────────────────
    FOLDER_TABLE_MAP: str = (
        "analista_contabil:kb_analista_contabil,"
        "analista_fiscal:kb_analista_fiscal,"
        "analista_departamento_pessoal:kb_analista_departamento_pessoal,"
        "analista_societario:kb_analista_societario,"
        "analista_abertura_empresas:kb_analista_abertura_empresas_ma"
    )

    # ── Pipeline ──────────────────────────────────────────────────────────────
    CHUNK_SIZE:    int = 1000
    CHUNK_OVERLAP: int = 200
    BATCH_SIZE:    int = 50

    # ── Crawler ───────────────────────────────────────────────────────────────
    AUTO_DOWNLOAD_ENABLED: bool = True
    AUTO_DOWNLOAD_CRON:    str  = "0 3 * * 1"

    # ── API ───────────────────────────────────────────────────────────────────
    API_SECRET_KEY: str = "change-me-in-railway"
    PORT:           int = 8000
    ALLOWED_ORIGINS: str = "*"

    class Config:
        env_file = ".env"
        extra    = "ignore"

    def get_folder_table_map(self) -> dict[str, str]:
        result = {}
        for pair in self.FOLDER_TABLE_MAP.split(","):
            pair = pair.strip()
            if ":" in pair:
                folder, table = pair.split(":", 1)
                result[folder.strip()] = table.strip()
        return result


settings = Settings()
