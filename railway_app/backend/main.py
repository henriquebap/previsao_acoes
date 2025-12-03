"""
Stock Predictor API - FastAPI Backend
Tech Challenge Fase 4 - FIAP Pos-Tech ML Engineering

Features:
- API RESTful para previsoes de acoes
- Persistencia em PostgreSQL
- Modelos LSTM do HuggingFace Hub
- WebSocket para atualizacoes em tempo real
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import os
from loguru import logger

from routes import predictions, stocks, websocket
from services.model_service import ModelService

# Database (opcional)
try:
    from database.service import get_db_service
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle: startup e shutdown."""
    # ==================== STARTUP ====================
    logger.info("🚀 Iniciando Stock Predictor API...")
    
    # Inicializar Database
    if DB_AVAILABLE:
        try:
            db = get_db_service()
            app.state.db = db
            logger.info("✅ PostgreSQL conectado e tabelas criadas")
        except Exception as e:
            logger.warning(f"⚠️ Database não disponível: {e}")
            app.state.db = None
    else:
        app.state.db = None
        logger.info("ℹ️ Rodando sem banco de dados (apenas cache)")
    
    # Inicializar Model Service
    model_service = ModelService()
    app.state.model_service = model_service
    
    # Pre-carregar modelo BASE
    try:
        model_service.get_model("BASE")
        logger.info("✅ Modelo BASE carregado")
    except Exception as e:
        logger.warning(f"⚠️ Erro ao carregar modelo BASE: {e}")
    
    logger.info("=" * 50)
    logger.info("🎯 API pronta para receber requisições!")
    logger.info("=" * 50)
    
    yield
    
    # ==================== SHUTDOWN ====================
    logger.info("👋 Encerrando API...")


# Criar app
app = FastAPI(
    title="Stock Predictor API",
    description="""
    API de previsão de preços de ações com LSTM.
    
    ## Features
    - 📈 Previsões de preços com modelos LSTM
    - 📊 Dados históricos de ações (Yahoo Finance)
    - 🗃️ Persistência em PostgreSQL
    - 🔄 WebSocket para preços em tempo real
    
    ## Modelos
    - **BASE**: Modelo genérico treinado com múltiplas ações
    - **Específicos**: Modelos especializados para ações populares (AAPL, GOOGL, NVDA)
    
    ## Tech Challenge Fase 4 - FIAP Pós-Tech ML Engineering
    """,
    version="2.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rotas
app.include_router(stocks.router, prefix="/api/stocks", tags=["Stocks"])
app.include_router(predictions.router, prefix="/api/predictions", tags=["Predictions"])
app.include_router(websocket.router, prefix="/ws", tags=["WebSocket"])


@app.get("/")
async def root():
    """Informações da API."""
    return {
        "name": "Stock Predictor API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "features": [
            "LSTM predictions",
            "PostgreSQL persistence",
            "HuggingFace Hub models",
            "Real-time WebSocket"
        ]
    }


@app.get("/health")
async def health():
    """Health check da API."""
    db_status = "connected" if (hasattr(app.state, 'db') and app.state.db) else "not_configured"
    model_status = "loaded" if (hasattr(app.state, 'model_service') and app.state.model_service) else "not_loaded"
    
    return {
        "status": "healthy",
        "database": db_status,
        "model_service": model_status,
        "timestamp": __import__('datetime').datetime.utcnow().isoformat()
    }


@app.get("/api/stats")
async def get_stats():
    """Estatísticas do sistema."""
    stats = {
        "api_version": "2.0.0",
        "database_available": hasattr(app.state, 'db') and app.state.db is not None
    }
    
    if hasattr(app.state, 'model_service'):
        stats["available_models"] = app.state.model_service.list_available_models()
    
    return stats


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
