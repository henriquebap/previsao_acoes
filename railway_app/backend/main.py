"""
Stock Predictor API - FastAPI Backend
Tech Challenge Fase 4 - FIAP Pos-Tech ML Engineering

Features:
- API RESTful para previsoes de acoes
- Persistencia em PostgreSQL
- Modelos LSTM do HuggingFace Hub
- WebSocket para atualizacoes em tempo real
- Monitoramento de performance em tempo real
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import os
import time
import asyncio
from datetime import datetime
from loguru import logger

from fastapi.responses import Response
from routes import predictions, stocks, websocket, ml_health
from services.model_service import ModelService
from services.monitoring import get_monitoring_service
from services.model_evaluation import get_evaluation_service
from services.prometheus_metrics import get_prometheus_metrics as get_prom_metrics

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
    logger.info("Iniciando Stock Predictor API...")
    
    # Inicializar Monitoring
    monitoring = get_monitoring_service()
    app.state.monitoring = monitoring
    logger.info(" Monitoramento inicializado")
    
    # Inicializar Database
    if DB_AVAILABLE:
        try:
            db = get_db_service()
            app.state.db = db
            logger.info(" PostgreSQL conectado e tabelas criadas")
        except Exception as e:
            logger.warning(f" Database não disponível: {e}")
            app.state.db = None
    else:
        app.state.db = None
        logger.info(" Rodando sem banco de dados (apenas cache)")
    
    # Inicializar Model Service
    model_service = ModelService()
    app.state.model_service = model_service
    
    # Pre-carregar modelo BASE
    try:
        model_service.get_model("BASE")
        logger.info(" Modelo BASE carregado")
    except Exception as e:
        logger.warning(f" Erro ao carregar modelo BASE: {e}")
    
    # Task para coletar métricas de sistema periodicamente
    async def collect_system_metrics():
        # #region agent log
        import json, time
        with open('/Users/henriquebap/Pessoal/PosTech/previsao_acoes/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"initial","hypothesisId":"H3","location":"main.py:73","message":"collect_system_metrics INICIADO","data":{"interval_seconds":30},"timestamp":int(time.time()*1000)})+'\n')
        # #endregion
        iteration = 0
        while True:
            # #region agent log
            import psutil
            mem = psutil.virtual_memory()
            with open('/Users/henriquebap/Pessoal/PosTech/previsao_acoes/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"initial","hypothesisId":"H3","location":"main.py:76","message":"Coletando métricas do sistema","data":{"iteration":iteration,"memory_percent":mem.percent,"memory_used_mb":mem.used/1024/1024,"cpu_percent":psutil.cpu_percent()},"timestamp":int(time.time()*1000)})+'\n')
            iteration += 1
            # #endregion
            monitoring.record_system_metrics()
            await asyncio.sleep(30)  # A cada 30 segundos
    
    # Iniciar coleta em background
    asyncio.create_task(collect_system_metrics())
    
    logger.info("=" * 50)
    logger.info(" API pronta para receber requisições!")
    logger.info(" Métricas disponíveis em /api/monitoring")
    logger.info("=" * 50)
    
    yield
    
    # ==================== SHUTDOWN ====================
    logger.info(" Encerrando API...")


# Criar app
app = FastAPI(
    title="Stock Predictor API",
    description="""
    API de previsão de preços de ações com LSTM.
    
    ## Features
    -  Previsões de preços com modelos LSTM
    -  Dados históricos de ações (Yahoo Finance)
    -  Persistência em PostgreSQL
    -  WebSocket para preços em tempo real
    
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


# Middleware de Monitoramento
@app.middleware("http")
async def monitoring_middleware(request: Request, call_next):
    """Middleware que registra métricas de cada requisição."""
    # #region agent log
    import json
    with open('/Users/henriquebap/Pessoal/PosTech/previsao_acoes/.cursor/debug.log', 'a') as f:
        f.write(json.dumps({"sessionId":"debug-session","runId":"initial","hypothesisId":"H5","location":"main.py:125","message":"Request recebida","data":{"path":request.url.path,"method":request.method},"timestamp":int(time.time()*1000)})+'\n')
    # #endregion
    start_time = time.time()
    
    # Processar requisição
    response = await call_next(request)
    
    # Calcular tempo de resposta
    duration_seconds = time.time() - start_time
    response_time_ms = duration_seconds * 1000
    
    # Registrar métricas internas
    monitoring = getattr(request.app.state, 'monitoring', None)
    path = request.url.path
    
    # Extrair symbol se for endpoint de previsão
    symbol = None
    if '/predictions/' in path:
        parts = path.split('/')
        for i, part in enumerate(parts):
            if part == 'predictions' and i + 1 < len(parts):
                symbol = parts[i + 1].upper()
                break
    
    if monitoring:
        monitoring.record_request(
            endpoint=path,
            method=request.method,
            status_code=response.status_code,
            response_time_ms=response_time_ms,
            symbol=symbol
        )
    
    # Registrar métricas Prometheus
    try:
        prom = get_prom_metrics()
        prom.record_request(
            method=request.method,
            endpoint=path,
            status=response.status_code,
            duration_seconds=duration_seconds
        )
    except Exception:
        pass  # Não falhar se Prometheus não disponível
    
    # Adicionar header com tempo de resposta
    response.headers["X-Response-Time-Ms"] = str(round(response_time_ms, 2))
    
    return response


# Rotas
app.include_router(stocks.router, prefix="/api/stocks", tags=["Stocks"])
app.include_router(predictions.router, prefix="/api/predictions", tags=["Predictions"])
app.include_router(websocket.router, prefix="/ws", tags=["WebSocket"])
app.include_router(ml_health.router, tags=["ML Health"])


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
    # #region agent log
    import json, time
    with open('/Users/henriquebap/Pessoal/PosTech/previsao_acoes/.cursor/debug.log', 'a') as f:
        f.write(json.dumps({"sessionId":"debug-session","runId":"initial","hypothesisId":"H5","location":"main.py:201","message":"Health check chamado","data":{},"timestamp":int(time.time()*1000)})+'\n')
    # #endregion
    
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


@app.get("/api/database/status")
async def get_database_status():
    """
    Status detalhado do banco de dados.
    
    Retorna:
    - Status da conexão
    - Lista de tabelas
    - Contagem de registros por tabela
    """
    db = getattr(app.state, 'db', None)
    
    if not db:
        return {
            "status": "not_configured",
            "message": "DATABASE_URL não está configurada"
        }
    
    return db.get_database_status()


@app.post("/api/database/migrate")
async def run_migration():
    """
    Força a criação/atualização das tabelas no banco.
    
    Útil se as tabelas não foram criadas automaticamente.
    """
    db = getattr(app.state, 'db', None)
    
    if not db:
        return {
            "status": "error",
            "message": "DATABASE_URL não está configurada"
        }
    
    try:
        db.create_tables()
        return {
            "status": "success",
            "message": "Tabelas criadas/atualizadas com sucesso",
            "details": db.get_database_status()
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


# ==================== MONITORING ENDPOINTS ====================

@app.get("/api/monitoring", tags=["Monitoring"])
async def get_monitoring_summary():
    """
    Resumo completo de métricas de monitoramento.
    
    Inclui:
    - Uptime da API
    - Total de requisições e taxa de erros
    - Métricas por endpoint (tempo médio, min, max)
    - Métricas de inferência do modelo
    - Utilização de CPU e memória
    """
    monitoring = getattr(app.state, 'monitoring', None)
    if not monitoring:
        return {"error": "Monitoring not available"}
    
    return monitoring.get_summary()


@app.get("/api/monitoring/requests", tags=["Monitoring"])
async def get_recent_requests(limit: int = 50):
    """
    Lista requisições recentes com métricas detalhadas.
    
    Args:
        limit: Número máximo de requisições a retornar (default: 50)
    """
    monitoring = getattr(app.state, 'monitoring', None)
    if not monitoring:
        return {"error": "Monitoring not available"}
    
    return {
        "requests": monitoring.get_recent_requests(limit),
        "total": len(monitoring.requests)
    }


@app.get("/api/monitoring/system", tags=["Monitoring"])
async def get_system_metrics():
    """
    Histórico de métricas de sistema (CPU, memória).
    
    Coleta automática a cada 30 segundos.
    """
    monitoring = getattr(app.state, 'monitoring', None)
    if not monitoring:
        return {"error": "Monitoring not available"}
    
    # Coletar métrica atual
    current = monitoring.record_system_metrics()
    
    return {
        "current": {
            "cpu_percent": current.cpu_percent if current else None,
            "memory_percent": current.memory_percent if current else None,
            "memory_used_mb": round(current.memory_used_mb, 2) if current else None,
            "memory_available_mb": round(current.memory_available_mb, 2) if current else None
        } if current else None,
        "history": monitoring.get_system_history()
    }


@app.get("/metrics", tags=["Monitoring"])
async def prometheus_metrics_endpoint():
    """
    Endpoint padrão Prometheus para scraping de métricas.
    
    Formato: Prometheus text exposition format
    
    Use este endpoint para integração com Prometheus/Grafana.
    Configure no prometheus.yml:
    ```yaml
    scrape_configs:
      - job_name: 'stock-predictor'
        static_configs:
          - targets: ['your-api-url:8000']
        metrics_path: '/metrics'
    ```
    """
    try:
        prom = get_prom_metrics()
        
        # Atualizar contagem de modelos carregados
        if hasattr(app.state, 'model_service'):
            models_loaded = len(app.state.model_service.model_cache)
            prom.set_models_loaded(models_loaded)
        
        metrics_data = prom.get_metrics()
        return Response(
            content=metrics_data,
            media_type=prom.get_content_type()
        )
    except Exception as e:
        logger.error(f"Erro ao gerar métricas Prometheus: {e}")
        return Response(
            content=f"# Error generating metrics: {e}",
            media_type="text/plain"
        )


@app.get("/api/monitoring/prometheus", tags=["Monitoring"])
async def get_prometheus_metrics_legacy():
    """
    Métricas em formato Prometheus (formato legado).
    
    Pode ser usado para integração com Grafana/Prometheus.
    Recomendado usar /metrics para scraping automático.
    """
    monitoring = getattr(app.state, 'monitoring', None)
    if not monitoring:
        return "# Monitoring not available"
    
    summary = monitoring.get_summary()
    
    lines = [
        "# HELP api_uptime_seconds Uptime da API em segundos",
        "# TYPE api_uptime_seconds gauge",
        f"api_uptime_seconds {summary['uptime_seconds']}",
        "",
        "# HELP api_requests_total Total de requisições",
        "# TYPE api_requests_total counter",
        f"api_requests_total {summary['total_requests']}",
        "",
        "# HELP api_errors_total Total de erros",
        "# TYPE api_errors_total counter",
        f"api_errors_total {summary['total_errors']}",
        "",
        "# HELP api_predictions_total Total de previsões",
        "# TYPE api_predictions_total counter",
        f"api_predictions_total {summary['total_predictions']}",
        "",
    ]
    
    # Métricas por endpoint
    lines.append("# HELP api_endpoint_requests_total Requisições por endpoint")
    lines.append("# TYPE api_endpoint_requests_total counter")
    for endpoint, stats in summary.get('endpoints', {}).items():
        safe_endpoint = endpoint.replace('/', '_').replace('{', '').replace('}', '')
        lines.append(f'api_endpoint_requests_total{{endpoint="{safe_endpoint}"}} {stats["count"]}')
    
    lines.append("")
    lines.append("# HELP api_endpoint_response_time_ms Tempo médio de resposta por endpoint")
    lines.append("# TYPE api_endpoint_response_time_ms gauge")
    for endpoint, stats in summary.get('endpoints', {}).items():
        safe_endpoint = endpoint.replace('/', '_').replace('{', '').replace('}', '')
        lines.append(f'api_endpoint_response_time_ms{{endpoint="{safe_endpoint}"}} {stats["avg_time_ms"]}')
    
    # Sistema
    if summary.get('system'):
        lines.append("")
        lines.append("# HELP system_cpu_percent Uso de CPU")
        lines.append("# TYPE system_cpu_percent gauge")
        lines.append(f"system_cpu_percent {summary['system']['cpu_percent']}")
        lines.append("")
        lines.append("# HELP system_memory_percent Uso de memória")
        lines.append("# TYPE system_memory_percent gauge")
        lines.append(f"system_memory_percent {summary['system']['memory_percent']}")
    
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(content="\n".join(lines), media_type="text/plain")


# ==================== MODEL EVALUATION ENDPOINTS ====================

@app.get("/api/model/evaluation", tags=["Model Evaluation"])
async def evaluate_model_performance(days: int = 7):
    """
    Avalia performance REAL do modelo em produção.
    
    Compara previsões passadas com os valores reais que ocorreram.
    
    Métricas:
    - MAPE: Erro percentual médio das previsões
    - Acurácia Direcional: % de vezes que acertou se sobe ou desce
    - Erro em dólares: Diferença média entre previsto e real
    
    Args:
        days: Período de avaliação em dias (default: 7)
    """
    db = getattr(app.state, 'db', None)
    if not db:
        return {"error": "Database não disponível para avaliação"}
    
    evaluation_service = get_evaluation_service(db)
    return evaluation_service.evaluate_past_predictions(days=days)


@app.get("/api/model/drift/{symbol}", tags=["Model Evaluation"])
async def check_data_drift(symbol: str, window_days: int = 30):
    """
    Detecta drift nos dados de uma ação.
    
    Drift = mudança significativa na distribuição dos dados.
    Quando há drift, o modelo pode precisar ser retreinado.
    
    Verifica:
    - Mudança no preço médio
    - Mudança na volatilidade
    - Mudança no volume
    
    Args:
        symbol: Ticker da ação (ex: AAPL)
        window_days: Período histórico para comparação (default: 30)
    """
    db = getattr(app.state, 'db', None)
    if not db:
        return {"error": "Database não disponível"}
    
    evaluation_service = get_evaluation_service(db)
    return evaluation_service.detect_data_drift(symbol.upper(), window_days)


@app.get("/api/model/health", tags=["Model Evaluation"])
async def model_health_check():
    """
    Health check completo do modelo em produção.
    
    Combina:
    - Avaliação de performance (últimos 7 dias)
    - Status de drift para ações populares
    - Recomendações automáticas
    """
    db = getattr(app.state, 'db', None)
    if not db:
        return {"error": "Database não disponível"}
    
    evaluation_service = get_evaluation_service(db)
    
    # Avaliar performance
    performance = evaluation_service.evaluate_past_predictions(days=7)
    
    # Verificar drift nas ações mais usadas
    popular_symbols = ["AAPL", "NVDA", "GOOGL", "MSFT"]
    drift_status = {}
    
    for symbol in popular_symbols:
        drift = evaluation_service.detect_data_drift(symbol)
        if drift.get("status") != "error":
            drift_status[symbol] = {
                "drift_detected": drift.get("drift_detected", False),
                "price_drift": drift.get("metrics", {}).get("price_drift_percent", 0)
            }
    
    # Determinar saúde geral
    has_performance_issues = performance.get("metrics", {}).get("mape_percent", 100) > 10
    has_drift_issues = any(d.get("drift_detected") for d in drift_status.values())
    
    if has_performance_issues or has_drift_issues:
        health_status = "WARNING"
        health_message = "Modelo precisa de atenção"
    else:
        health_status = "HEALTHY"
        health_message = "Modelo funcionando normalmente"
    
    return {
        "status": health_status,
        "message": health_message,
        "performance": performance.get("metrics") if performance.get("status") == "success" else None,
        "quality": performance.get("quality_assessment") if performance.get("status") == "success" else None,
        "drift_status": drift_status,
        "recommendations": [
            performance.get("quality_assessment", {}).get("recommendation", "Sem dados suficientes")
        ] if performance.get("status") == "success" else ["Aguardando mais previsões para avaliar"],
        "timestamp": datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
