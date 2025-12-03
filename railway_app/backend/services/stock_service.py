"""
Stock Service - Wrapper para reutilizar src/data/data_loader
Principio DRY: Importa codigo existente ao inves de duplicar
"""
import sys
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime, timedelta
import pandas as pd
import time

# Adicionar raiz do projeto ao path para importar src/
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_loader import StockDataLoader


class StockService:
    """
    Servico para obter dados de acoes.
    Reutiliza StockDataLoader de src/data/
    """
    
    # Cache de dados (em producao usar Redis)
    _cache: Dict[str, dict] = {}
    _cache_ttl = 300  # 5 minutos
    
    # Mapeamento de nomes de empresas
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
        "GS": "Goldman Sachs Group",
        "WMT": "Walmart Inc.",
        "KO": "The Coca-Cola Company",
        "MCD": "McDonald's Corporation",
        "NKE": "Nike Inc.",
        "DIS": "The Walt Disney Company",
        "JNJ": "Johnson & Johnson",
        "PFE": "Pfizer Inc.",
        "PETR4.SA": "Petrobras",
        "VALE3.SA": "Vale S.A.",
        "ITUB4.SA": "Itau Unibanco",
        "BBDC4.SA": "Bradesco",
        "ABEV3.SA": "Ambev S.A.",
        "WEGE3.SA": "WEG S.A.",
        "MGLU3.SA": "Magazine Luiza",
        "NU": "Nubank",
        "MELI": "MercadoLibre Inc.",
    }
    
    def __init__(self):
        # Reutilizar StockDataLoader de src/
        self.loader = StockDataLoader()
    
    def _get_cache_key(self, symbol: str, days: int) -> str:
        return f"{symbol}_{days}"
    
    def _is_cache_valid(self, key: str) -> bool:
        if key not in self._cache:
            return False
        cached = self._cache[key]
        return (time.time() - cached['timestamp']) < self._cache_ttl
    
    def get_stock_data(self, symbol: str, days: int = 365) -> Optional[pd.DataFrame]:
        """
        Obtem dados historicos de uma acao usando src/data/data_loader.
        
        Args:
            symbol: Ticker da acao (ex: AAPL, PETR4.SA)
            days: Numero de dias de historico
        
        Returns:
            DataFrame com dados ou None se falhar
        """
        symbol = symbol.upper()
        cache_key = self._get_cache_key(symbol, days)
        
        # Verificar cache
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]['data'].copy()
        
        end = datetime.now()
        start = end - timedelta(days=days + 30)  # Extra para garantir
        
        try:
            # Usar StockDataLoader de src/
            df = self.loader.load_stock_data(
                symbol,
                start.strftime('%Y-%m-%d'),
                end.strftime('%Y-%m-%d')
            )
            
            if df.empty:
                return None
            
            # Validar dados
            self.loader.validate_data(df)
            
            # Limitar ao numero de dias solicitado
            df = df.tail(days)
            
            # Cachear
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
        symbol = symbol.upper()
        
        if symbol in self.COMPANY_NAMES:
            return self.COMPANY_NAMES[symbol]
        
        return symbol
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Obtem preco atual."""
        df = self.get_stock_data(symbol, days=5)
        if df is not None and len(df) > 0:
            return float(df['close'].iloc[-1])
        return None
    
    def get_price_change(self, symbol: str) -> Optional[dict]:
        """Obtem variacao de preco."""
        df = self.get_stock_data(symbol, days=30)
        if df is None or len(df) < 2:
            return None
        
        current = float(df['close'].iloc[-1])
        prev_day = float(df['close'].iloc[-2])
        week_ago = float(df['close'].iloc[-5]) if len(df) >= 5 else prev_day
        month_ago = float(df['close'].iloc[0])
        
        return {
            'current': current,
            'day_change': ((current - prev_day) / prev_day) * 100,
            'week_change': ((current - week_ago) / week_ago) * 100,
            'month_change': ((current - month_ago) / month_ago) * 100
        }
    
    def clear_cache(self, symbol: Optional[str] = None):
        """Limpa cache."""
        if symbol:
            keys_to_remove = [k for k in self._cache if k.startswith(symbol)]
            for key in keys_to_remove:
                del self._cache[key]
        else:
            self._cache.clear()
