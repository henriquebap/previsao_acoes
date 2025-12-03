"""
Predictions Routes - Previsões LSTM
"""
from fastapi import APIRouter, HTTPException, Query, Request
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
import numpy as np

from services.stock_service import StockService
from services.model_service import ModelService


router = APIRouter()
stock_service = StockService()


class PredictionResponse(BaseModel):
    symbol: str
    current_price: float
    predicted_price: float
    change_percent: float
    direction: str
    confidence: str
    model_type: str
    indicators: dict
    timestamp: str


prediction_history: List[dict] = []


@router.get("/{symbol}", response_model=PredictionResponse)
async def get_prediction(request: Request, symbol: str):
    """Obtém previsão LSTM para uma ação."""
    from routes.stocks import resolve_symbol
    resolved_symbol = resolve_symbol(symbol)
    
    try:
        df = stock_service.get_stock_data(resolved_symbol, days=400)
        
        if df is None or len(df) < 70:
            raise HTTPException(status_code=400, detail=f"Dados insuficientes para {resolved_symbol}")
        
        current_price = float(df['close'].iloc[-1])
        model_service: ModelService = request.app.state.model_service
        prediction_result = model_service.predict(resolved_symbol, df)
        
        predicted_price = prediction_result['predicted_price']
        change_percent = ((predicted_price - current_price) / current_price) * 100
        
        if change_percent > 2:
            direction, confidence = "📈 ALTA FORTE", "Alta"
        elif change_percent > 0.5:
            direction, confidence = "📈 ALTA", "Moderada"
        elif change_percent < -2:
            direction, confidence = "📉 BAIXA FORTE", "Alta"
        elif change_percent < -0.5:
            direction, confidence = "📉 BAIXA", "Moderada"
        else:
            direction, confidence = "➡️ LATERAL", "Baixa"
        
        ma_7 = float(df['close'].rolling(7).mean().iloc[-1])
        ma_30 = float(df['close'].rolling(30).mean().iloc[-1])
        
        return PredictionResponse(
            symbol=resolved_symbol,
            current_price=round(current_price, 2),
            predicted_price=round(predicted_price, 2),
            change_percent=round(change_percent, 2),
            direction=direction,
            confidence=confidence,
            model_type=prediction_result['model_type'],
            indicators={"ma_7": round(ma_7, 2), "ma_30": round(ma_30, 2), "trend": "bullish" if ma_7 > ma_30 else "bearish"},
            timestamp=datetime.now().isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro: {str(e)}")


@router.get("/history/recent")
async def get_history(limit: int = Query(default=20, ge=1, le=100)):
    """Histórico de previsões."""
    return {"history": prediction_history[-limit:][::-1], "total": len(prediction_history)}

