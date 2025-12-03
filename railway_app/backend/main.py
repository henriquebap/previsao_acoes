"""
Stock Predictor API - FastAPI Backend
Tech Challenge Fase 4 - FIAP Pós-Tech ML Engineering
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from routes import predictions, stocks, websocket
from services.model_service import ModelService


# Lifespan para inicialização
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: pré-carregar modelos
    print("🚀 Iniciando Stock Predictor API...")
    model_service = ModelService()
    app.state.model_service = model_service
    
    # Pré-carregar modelo base
    try:
        model_service.get_model("BASE")
        print("✅ Modelo BASE carregado")
    except Exception as e:
        print(f"⚠️ Erro ao carregar modelo BASE: {e}")
    
    yield
    
    # Shutdown
    print("👋 Encerrando API...")


# Criar app
app = FastAPI(
    title="Stock Predictor API",
    description="API de previsão de preços de ações com LSTM",
    version="1.0.0",
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
    return {
        "name": "Stock Predictor API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

