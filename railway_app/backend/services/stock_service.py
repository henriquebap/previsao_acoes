"""
Stock Service - Dados de acoes do Yahoo Finance
Deploy auto-contido para Railway
"""
from typing import Optional, Dict
from datetime import datetime, timedelta
import pandas as pd
import time

# Importar de core/ (copia local para deploy)
from core.data_loader import StockDataLoader


class StockService:
    """Servico para obter dados de acoes."""
    
    _cache: Dict[str, dict] = {}
    _cache_ttl = 300  # 5 minutos
    
    COMPANY_NAMES = {
        "AAPL": "Apple Inc.",
        "GOOGL": "Alphabet Inc.",
        "MSFT": "Microsoft Corporation",
        "AMZN": "Amazon.com Inc.",
        "META": "Meta Platforms Inc.",
        "NVDA": "NVIDIA Corporation",
        "TSLA": "Tesla Inc.",
        "NFLX": "Netflix Inc.",
        "JPM": "JPMorgan Chase & Co.",
        "BAC": "Bank of America Corp.",
        "V": "Visa Inc.",
        "MA": "Mastercard Inc.",
        "WMT": "Walmart Inc.",
        "KO": "The Coca-Cola Company",
        "DIS": "The Walt Disney Company",
        "JNJ": "Johnson & Johnson",
        "PFE": "Pfizer Inc.",
        "PETR4.SA": "Petrobras",
        "VALE3.SA": "Vale S.A.",
        "ITUB4.SA": "Itau Unibanco",
        "BBDC4.SA": "Bradesco",
        "ABEV3.SA": "Ambev S.A.",
        "NU": "Nubank",
        "MELI": "MercadoLibre Inc.",
    }
    
    def __init__(self):
        self.loader = StockDataLoader()
    
    def _get_cache_key(self, symbol: str, days: int) -> str:
        return f"{symbol}_{days}"
    
    def _is_cache_valid(self, key: str) -> bool:
        if key not in self._cache:
            return False
        cached = self._cache[key]
        return (time.time() - cached['timestamp']) < self._cache_ttl
    
    def get_stock_data(self, symbol: str, days: int = 365) -> Optional[pd.DataFrame]:
        """Obtem dados historicos de uma acao."""
        symbol = symbol.upper()
        cache_key = self._get_cache_key(symbol, days)
        
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]['data'].copy()
        
        end = datetime.now()
        start = end - timedelta(days=days + 30)
        
        try:
            df = self.loader.load_stock_data(
                symbol,
                start.strftime('%Y-%m-%d'),
                end.strftime('%Y-%m-%d')
            )
            
            if df.empty:
                return None
            
            self.loader.validate_data(df)
            df = df.tail(days)
            
            self._cache[cache_key] = {
                'data': df,
                'timestamp': time.time()
            }
            
            return df.copy()
            
        except Exception as e:
            print(f"Erro ao obter dados de {symbol}: {e}")
            return None
    
    def get_company_name(self, symbol: str) -> str:
        """Obtem nome da empresa pelo ticker."""
        return self.COMPANY_NAMES.get(symbol.upper(), symbol)
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Obtem preco atual."""
        df = self.get_stock_data(symbol, days=5)
        if df is not None and len(df) > 0:
            return float(df['close'].iloc[-1])
        return None
    
    def clear_cache(self, symbol: Optional[str] = None):
        """Limpa cache."""
        if symbol:
            keys_to_remove = [k for k in self._cache if k.startswith(symbol)]
            for key in keys_to_remove:
                del self._cache[key]
        else:
            self._cache.clear()
