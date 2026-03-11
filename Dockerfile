# Use Python 3.11 slim as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed by chromadb and torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (so Docker caches this layer — only re-runs if requirements change)
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Pre-download the Bulgarian embedding model into the image
# (so the container doesn't need to download it on every cold start)
RUN python -c "\
from transformers import AutoTokenizer, AutoModel; \
AutoTokenizer.from_pretrained('rmihaylov/roberta-base-nli-stsb-bg'); \
AutoModel.from_pretrained('rmihaylov/roberta-base-nli-stsb-bg'); \
print('Model downloaded.')"

# Build the ChromaDB vector database from the listings
# (runs ingest.py so the DB is baked into the image)
RUN python rag/ingest.py

# Cloud Run injects a PORT environment variable (default 8080)
# The CMD uses shell form so ${PORT} gets expanded at runtime
CMD uvicorn web.app:app --host 0.0.0.0 --port ${PORT:-8080}
