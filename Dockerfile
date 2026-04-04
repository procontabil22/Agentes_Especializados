# FinTax Agents — Dockerfile para Railway
# Estrutura flat: todos os .py na raiz do repositório

FROM ubuntu:22.04

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

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

RUN python3 -m pip install --upgrade pip --quiet

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN playwright install chromium \
    && playwright install-deps chromium

COPY . .

RUN mkdir -p /tmp/fintax_downloads

EXPOSE 8000

# "main:app" → procura main.py na raiz de /app (WORKDIR)
# shell form garante expansão de $PORT pelo sh
CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"
