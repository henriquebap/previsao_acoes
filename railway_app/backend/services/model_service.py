"""
Model Service - Gerenciamento de modelos LSTM
Deploy auto-contido para Railway
"""
from pathlib import Path
from typing import Dict, Optional, List
import pandas as pd
import numpy as np
from loguru import logger

# Importar de core/ (copia local para deploy)
from core.lstm_model import LSTMPredictor
from core.preprocessor import StockDataPreprocessor

# HuggingFace Hub para baixar modelos
from huggingface_hub import hf_hub_download


class ModelService:
    """Servico de gerenciamento de modelos LSTM."""
    
    HUB_REPO = "henriquebap/stock-predictor-lstm"
    LOCAL_CACHE = Path("models")
    
    def __init__(self):
        self.model_cache: Dict[str, dict] = {}
        self.LOCAL_CACHE.mkdir(exist_ok=True)
        logger.info(f"🧠 ModelService inicializado | Hub: {self.HUB_REPO}")
    
    def _download_from_hub(self, filename: str) -> Path:
        """Baixa arquivo do HuggingFace Hub."""
        logger.info(f"📥 Baixando do Hub: {filename}")
        path = Path(hf_hub_download(
            repo_id=self.HUB_REPO,
            filename=filename,
            cache_dir=str(self.LOCAL_CACHE / "hub_cache")
        ))
        logger.info(f"✅ Download concluido: {filename}")
        return path
    
    def _load_model(self, symbol: str) -> Optional[dict]:
        """Carrega modelo do cache local ou Hub."""
        model_file = f"lstm_model_{symbol}.pth"
        scaler_file = f"scaler_{symbol}.pkl"
        
        logger.info(f"🔍 Procurando modelo para {symbol}...")
        
        # Tentar cache local primeiro
        local_model = self.LOCAL_CACHE / model_file
        local_scaler = self.LOCAL_CACHE / scaler_file
        
        if local_model.exists() and local_scaler.exists():
            model_path = local_model
            scaler_path = local_scaler
            source = "local"
            logger.info(f"📁 Modelo LOCAL encontrado para {symbol}")
        else:
            # Tentar HuggingFace Hub - modelo especifico
            logger.info(f"🌐 Buscando modelo {symbol} no HuggingFace Hub...")
            try:
                model_path = self._download_from_hub(model_file)
                scaler_path = self._download_from_hub(scaler_file)
                source = "hub_specific"
                logger.info(f"✅ Modelo ESPECÍFICO para {symbol} encontrado no Hub!")
            except Exception as e:
                # Fallback para modelo BASE
                logger.warning(f"⚠️ Modelo específico para {symbol} não encontrado: {e}")
                logger.info(f"🔄 Usando modelo BASE genérico...")
                try:
                    model_path = self._download_from_hub("lstm_model_BASE.pth")
                    scaler_path = self._download_from_hub("scaler_BASE.pkl")
                    source = "hub_base"
                    logger.info(f"✅ Modelo BASE carregado para {symbol}")
                except Exception as e2:
                    logger.error(f"❌ Falha ao carregar modelo BASE: {e2}")
                    return None
        
        # Carregar modelo e preprocessor
        try:
            model = LSTMPredictor.load(model_path)
            preprocessor = StockDataPreprocessor.load(scaler_path)
            
            logger.info(f"🎯 Modelo carregado | Symbol: {symbol} | Source: {source}")
            
            return {
                'model': model,
                'preprocessor': preprocessor,
                'source': source,
                'symbol_requested': symbol
            }
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo: {e}")
            return None
    
    def get_model(self, symbol: str) -> Optional[dict]:
        """Obtem modelo do cache ou carrega."""
        symbol = symbol.upper()
        
        if symbol not in self.model_cache:
            logger.info(f"📦 Modelo {symbol} não está em cache, carregando...")
            model_data = self._load_model(symbol)
            if model_data:
                self.model_cache[symbol] = model_data
        else:
            logger.info(f"⚡ Modelo {symbol} encontrado em cache!")
        
        return self.model_cache.get(symbol)
    
    def predict(self, symbol: str, df: pd.DataFrame) -> dict:
        """Faz previsao para um simbolo."""
        logger.info(f"🔮 Iniciando previsão para {symbol}...")
        
        # Tentar modelo especifico, depois BASE
        model_data = self.get_model(symbol)
        
        if not model_data:
            logger.warning(f"⚠️ Nenhum modelo disponível, tentando BASE...")
            model_data = self.get_model("BASE")
        
        if not model_data:
            # Fallback: media movel simples
            logger.warning(f"⚠️ Usando FALLBACK (média móvel) para {symbol}")
            current = float(df['close'].iloc[-1])
            momentum = float((df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5])
            predicted = current * (1 + momentum * 0.3)
            
            return {
                'predicted_price': predicted,
                'model_type': '⚠️ Fallback (MA)'
            }
        
        model: LSTMPredictor = model_data['model']
        preprocessor: StockDataPreprocessor = model_data['preprocessor']
        source = model_data['source']
        
        # Fazer previsao
        try:
            X = preprocessor.transform_for_prediction(df)
            predictions = model.predict(X)
            pred_scaled = predictions[0]
            predicted_price = preprocessor.inverse_transform_target(pred_scaled)
            
            # Determinar tipo do modelo para exibição
            if source == "hub_specific":
                model_type = f"🎯 LSTM Fine-tuned ({symbol})"
            elif source == "hub_base":
                model_type = f"🧠 LSTM Base (genérico)"
            elif source == "local":
                model_type = f"📁 LSTM Local ({symbol})"
            else:
                model_type = f"LSTM ({source})"
            
            logger.info(f"✅ Previsão concluída | {symbol} | ${predicted_price:.2f} | {model_type}")
            
            return {
                'predicted_price': float(predicted_price),
                'model_type': model_type
            }
        except Exception as e:
            logger.error(f"❌ Erro na previsão para {symbol}: {e}")
            current = float(df['close'].iloc[-1])
            return {
                'predicted_price': current,
                'model_type': f'⚠️ Erro: {str(e)[:30]}'
            }
    
    def list_available_models(self) -> List[str]:
        """Lista modelos disponiveis."""
        models = ["BASE"]
        
        for f in self.LOCAL_CACHE.glob("lstm_model_*.pth"):
            name = f.stem.replace("lstm_model_", "")
            if name not in models:
                models.append(name)
        
        # Modelos conhecidos no Hub
        hub_models = ["AAPL", "GOOGL", "NVDA"]
        for m in hub_models:
            if m not in models:
                models.append(m)
        
        return models
