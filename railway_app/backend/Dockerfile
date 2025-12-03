# Backend FastAPI Dockerfile
# Reutiliza codigo de src/ (principio DRY)
FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias do sistema
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar
COPY railway_app/backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copiar codigo fonte (src/ para reutilizacao)
COPY src/ ./src/
COPY config/ ./config/

# Copiar backend
COPY railway_app/backend/ ./railway_app/backend/

# Criar diretorio de modelos
RUN mkdir -p models

# Definir PYTHONPATH
ENV PYTHONPATH=/app

# Diretorio de trabalho do backend
WORKDIR /app/railway_app/backend

# Porta
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
