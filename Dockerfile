# FinTax Agents — Dockerfile para Railway
# Base: Ubuntu 22.04 (Jammy) — suportado oficialmente pelo Playwright

FROM ubuntu:22.04

# Evita prompts interativos durante apt-get
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ── Python 3.11 + dependências de sistema ────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3.11-distutils \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-por \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Garante que python3 aponta para 3.11
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Atualiza pip
RUN python3 -m pip install --upgrade pip --quiet

# ── Diretório de trabalho ─────────────────────────────────────────────────────
WORKDIR /app

# ── Dependências Python (camada cacheável) ────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Playwright — Ubuntu 22.04 é suportado oficialmente ───────────────────────
RUN playwright install chromium \
    && playwright install-deps chromium

# ── Código da aplicação ───────────────────────────────────────────────────────
COPY . .

# ── Diretório temporário para downloads ──────────────────────────────────────
RUN mkdir -p /tmp/fintax_downloads

# ── Porta exposta (Railway sobrescreve com $PORT) ─────────────────────────────
EXPOSE 8000

# ── Comando de start ──────────────────────────────────────────────────────────
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
