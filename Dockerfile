# FinTax Agents — Dockerfile para Railway
# Python 3.11 slim com Docling (OCR) + Playwright (portais JSF)

FROM python:3.11-slim

# ── Dependências de sistema ───────────────────────────────────────────────────
# Docling precisa: poppler (PDF rendering) + tesseract (OCR em PDFs escaneados)
# Playwright precisa: chromium deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-por \
    libgl1 \
    libglib2.0-0 \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# ── Diretório de trabalho ─────────────────────────────────────────────────────
WORKDIR /app

# ── Dependências Python (camada cacheável) ────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Playwright — instala Chromium após pip ────────────────────────────────────
RUN playwright install chromium --with-deps

# ── Código da aplicação ───────────────────────────────────────────────────────
COPY . .

# ── Diretório temporário para downloads ──────────────────────────────────────
RUN mkdir -p /tmp/fintax_downloads

# ── Porta exposta (Railway sobrescreve com $PORT) ─────────────────────────────
EXPOSE 8000

# ── Comando de start ──────────────────────────────────────────────────────────
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
