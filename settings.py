"""
config/settings.py
Centraliza todas as variáveis de ambiente do projeto.
Configure no Railway: Settings > Variables
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):

    # ── OpenAI ────────────────────────────────────────────────────────────────
    OPENAI_API_KEY: str
    EMBEDDING_MODEL: str = "text-embedding-3-small"   # 1536 dims
    EMBEDDING_DIMENSIONS: int = 1536

    # ── Anthropic (Claude) — agentes de chat ──────────────────────────────────
    ANTHROPIC_API_KEY: Optional[str] = None

    # ── Supabase ──────────────────────────────────────────────────────────────
    SUPABASE_URL: str
    SUPABASE_SERVICE_KEY: str          # service_role key (não anon)

    # ── Google Drive ──────────────────────────────────────────────────────────
    GOOGLE_CREDENTIALS_JSON: str       # JSON da Service Account (raw ou base64)
    GDRIVE_ROOT_FOLDER_ID: str         # ID da pasta raiz no Drive

    # ── Mapeamento pasta Drive → tabela Supabase (5 agentes) ──────────────────
    # Nomes corrigidos para os nomes reais das tabelas no Supabase
    FOLDER_TABLE_MAP: str = (
        "contabil:kb_analista_contabil,"
        "fiscal:kb_analista_fiscal,"
        "pessoal:kb_analista_departamento_pessoal,"
        "societario:kb_analista_societario,"
        "abertura_ma:kb_analista_abertura_empresas_ma"
    )

    # ── Pipeline ──────────────────────────────────────────────────────────────
    CHUNK_SIZE: int = 1000             # tokens por chunk
    CHUNK_OVERLAP: int = 200           # overlap para preservar contexto
    BATCH_SIZE: int = 50               # embeddings por batch na OpenAI

    # ── Crawler ───────────────────────────────────────────────────────────────
    AUTO_DOWNLOAD_ENABLED: bool = True
    AUTO_DOWNLOAD_CRON: str = "0 3 * * 1"   # toda segunda-feira às 3h

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
