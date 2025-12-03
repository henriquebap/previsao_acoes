"""
Model Service - Gerenciamento de modelos LSTM
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Optional, List
from huggingface_hub import hf_hub_download
import os


class LSTMModel(nn.Module):
    """Arquitetura LSTM para previsão."""
    
    def __init__(self, input_size=16, hidden_size=50, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])
        return self.fc(out)


class ModelService:
    """Serviço de gerenciamento de modelos."""
    
    HUB_REPO = "henriquebap/stock-predictor-lstm"
    LOCAL_CACHE = Path("models")
    SEQUENCE_LENGTH = 60
    
    def __init__(self):
        self.model_cache: Dict[str, dict] = {}
        self.LOCAL_CACHE.mkdir(exist_ok=True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def _download_from_hub(self, filename: str) -> str:
        """Baixa arquivo do HuggingFace Hub."""
        return hf_hub_download(
            repo_id=self.HUB_REPO,
            filename=filename,
            cache_dir=str(self.LOCAL_CACHE / "hub_cache")
        )
    
    def _load_model(self, symbol: str) -> Optional[dict]:
        """Carrega modelo do cache local ou Hub."""
        model_file = f"lstm_model_{symbol}.pth"
        scaler_file = f"scaler_{symbol}.pkl"
        
        # Tentar cache local primeiro
        local_model = self.LOCAL_CACHE / model_file
        local_scaler = self.LOCAL_CACHE / scaler_file
        
        if local_model.exists() and local_scaler.exists():
            model_path = str(local_model)
            scaler_path = str(local_scaler)
            source = "local"
        else:
            # Tentar HuggingFace Hub
            try:
                model_path = self._download_from_hub(model_file)
                scaler_path = self._download_from_hub(scaler_file)
                source = "hub"
            except:
                # Fallback para modelo BASE
                try:
                    model_path = self._download_from_hub("lstm_model_BASE.pth")
                    scaler_path = self._download_from_hub("scaler_BASE.pkl")
                    source = "base"
                except Exception as e:
                    print(f"Erro ao carregar modelo: {e}")
                    return None
        
        # Carregar checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Criar modelo
        model = LSTMModel(
            input_size=checkpoint.get('input_size', 16),
            hidden_size=checkpoint.get('hidden_size', 50),
            num_layers=checkpoint.get('num_layers', 2),
            dropout=checkpoint.get('dropout', 0.2)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        # Carregar scaler
        scaler_data = joblib.load(scaler_path)
        
        return {
            'model': model,
            'scaler': scaler_data.get('scaler'),
            'target_scaler': scaler_data.get('target_scaler'),
            'feature_columns': scaler_data.get('feature_columns', []),
            'source': source
        }
    
    def get_model(self, symbol: str) -> Optional[dict]:
        """Obtém modelo do cache ou carrega."""
        symbol = symbol.upper()
        
        if symbol not in self.model_cache:
            model_data = self._load_model(symbol)
            if model_data:
                self.model_cache[symbol] = model_data
        
        return self.model_cache.get(symbol)
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features técnicas."""
        df = df.copy()
        
        df['price_change'] = df['close'].pct_change().fillna(0)
        df['high_low_pct'] = ((df['high'] - df['low']) / df['low'].replace(0, 1)).fillna(0)
        df['close_open_pct'] = ((df['close'] - df['open']) / df['open'].replace(0, 1)).fillna(0)
        
        df['ma_7'] = df['close'].rolling(7, min_periods=1).mean()
        df['ma_30'] = df['close'].rolling(30, min_periods=1).mean()
        df['ma_90'] = df['close'].rolling(90, min_periods=1).mean()
        
        df['volatility_7'] = df['close'].rolling(7, min_periods=1).std().fillna(0)
        df['volatility_30'] = df['close'].rolling(30, min_periods=1).std().fillna(0)
        
        df['volume_change'] = df['volume'].pct_change().fillna(0)
        df['volume_ma_7'] = df['volume'].rolling(7, min_periods=1).mean()
        
        df['momentum'] = (df['close'] - df['close'].shift(4)).fillna(0)
        
        df = df.replace([np.inf, -np.inf], 0)
        return df.bfill().ffill()
    
    def predict(self, symbol: str, df: pd.DataFrame) -> dict:
        """Faz previsão para um símbolo."""
        # Tentar modelo específico, depois BASE
        model_data = self.get_model(symbol)
        
        if not model_data:
            model_data = self.get_model("BASE")
        
        if not model_data:
            # Fallback: média móvel simples
            current = float(df['close'].iloc[-1])
            ma_7 = float(df['close'].rolling(7).mean().iloc[-1])
            momentum = float((df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5])
            
            predicted = current * (1 + momentum * 0.3)
            
            return {
                'predicted_price': predicted,
                'model_type': 'Fallback (MA)'
            }
        
        model = model_data['model']
        scaler = model_data['scaler']
        target_scaler = model_data['target_scaler']
        feature_columns = model_data['feature_columns']
        source = model_data['source']
        
        # Preparar features
        df_feat = self._create_features(df)
        
        # Garantir que todas as colunas existem
        for col in feature_columns:
            if col not in df_feat.columns:
                df_feat[col] = 0
        
        features = df_feat[feature_columns].values
        features_scaled = scaler.transform(features)
        
        # Criar sequência
        X = features_scaled[-self.SEQUENCE_LENGTH:].reshape(1, self.SEQUENCE_LENGTH, len(feature_columns))
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Previsão
        with torch.no_grad():
            pred_scaled = model(X_tensor).cpu().numpy()[0, 0]
        
        # Inverter escala
        predicted_price = target_scaler.inverse_transform([[pred_scaled]])[0, 0]
        
        model_type = f"LSTM ({source.capitalize()})"
        if source == "base":
            model_type = "LSTM (Base Genérico)"
        elif source == "hub":
            model_type = f"LSTM (Hub - {symbol})"
        elif source == "local":
            model_type = f"LSTM (Local - {symbol})"
        
        return {
            'predicted_price': float(predicted_price),
            'model_type': model_type
        }
    
    def list_available_models(self) -> List[str]:
        """Lista modelos disponíveis."""
        models = ["BASE"]
        
        # Verificar locais
        for f in self.LOCAL_CACHE.glob("lstm_model_*.pth"):
            name = f.stem.replace("lstm_model_", "")
            if name not in models:
                models.append(name)
        
        # Adicionar conhecidos do Hub
        hub_models = ["AAPL", "GOOGL", "NVDA"]
        for m in hub_models:
            if m not in models:
                models.append(m)
        
        return models

