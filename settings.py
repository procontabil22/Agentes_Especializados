"""
settings.py
Centraliza todas as variáveis de ambiente do projeto.
Configure no Railway: Settings > Variables
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):

    # ── OpenAI ────────────────────────────────────────────────────────────────
    OPENAI_API_KEY: Optional[str] = None
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSIONS: int = 1536

    # ── LLMs de chat (configure ao menos um) ─────────────────────────────────
    # Ordem de prioridade: Anthropic → OpenAI → Gemini → Grok → DeepSeek
    ANTHROPIC_API_KEY: Optional[str] = None   # Claude
    GEMINI_API_KEY:    Optional[str] = None   # Google Gemini
    GROK_API_KEY:      Optional[str] = None   # xAI Grok
    DEEPSEEK_API_KEY:  Optional[str] = None   # DeepSeek

    # ── Supabase ──────────────────────────────────────────────────────────────
    SUPABASE_URL: str
    SUPABASE_SERVICE_KEY: str

    # ── Google Drive ──────────────────────────────────────────────────────────
    GOOGLE_CREDENTIALS_JSON: str
    GDRIVE_ROOT_FOLDER_ID: str

    # ── Mapeamento pasta Drive → tabela Supabase (5 agentes) ──────────────────
    FOLDER_TABLE_MAP: str = (
        "contabil:kb_analista_contabil,"
        "fiscal:kb_analista_fiscal,"
        "pessoal:kb_analista_departamento_pessoal,"
        "societario:kb_analista_societario,"
        "abertura_ma:kb_analista_abertura_empresas_ma"
    )

    # ── Pipeline ──────────────────────────────────────────────────────────────
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    BATCH_SIZE: int = 50

    # ── Crawler ───────────────────────────────────────────────────────────────
    AUTO_DOWNLOAD_ENABLED: bool = True
    AUTO_DOWNLOAD_CRON: str = "0 3 * * 1"

    # ── API ───────────────────────────────────────────────────────────────────
    API_SECRET_KEY: str = "change-me-in-railway"
    PORT: int = 8000

    class Config:
        env_file = ".env"
        extra = "ignore"

    def get_folder_table_map(self) -> dict[str, str]:
        result = {}
        for pair in self.FOLDER_TABLE_MAP.split(","):
            pair = pair.strip()
            if ":" in pair:
                folder, table = pair.split(":", 1)
                result[folder.strip()] = table.strip()
        return result


settings = Settings()
