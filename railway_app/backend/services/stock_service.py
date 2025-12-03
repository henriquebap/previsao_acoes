"""
Stock Service - Serviço de dados do Yahoo Finance
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict
import time


class StockService:
    """Serviço para obter dados de ações."""
    
    # Cache de dados (em produção usar Redis)
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
        "ITUB4.SA": "Itaú Unibanco",
        "BBDC4.SA": "Bradesco",
        "ABEV3.SA": "Ambev S.A.",
        "WEGE3.SA": "WEG S.A.",
        "MGLU3.SA": "Magazine Luiza",
        "NU": "Nubank",
        "MELI": "MercadoLibre Inc.",
    }
    
    def __init__(self):
        pass
    
    def _get_cache_key(self, symbol: str, days: int) -> str:
        return f"{symbol}_{days}"
    
    def _is_cache_valid(self, key: str) -> bool:
        if key not in self._cache:
            return False
        cached = self._cache[key]
        return (time.time() - cached['timestamp']) < self._cache_ttl
    
    def get_stock_data(self, symbol: str, days: int = 365) -> Optional[pd.DataFrame]:
        """
        Obtém dados históricos de uma ação.
        
        Args:
            symbol: Ticker da ação (ex: AAPL, PETR4.SA)
            days: Número de dias de histórico
        
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
            # Usar yfinance.download (mais estável que Ticker.history)
            df = yf.download(
                symbol,
                start=start.strftime('%Y-%m-%d'),
                end=end.strftime('%Y-%m-%d'),
                progress=False,
                auto_adjust=True
            )
            
            if df.empty:
                return None
            
            # Tratar MultiIndex
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df = df.reset_index()
            df.columns = df.columns.str.lower()
            
            # Renomear coluna de data
            for col in ['date', 'Date', 'datetime', 'Datetime']:
                if col.lower() in [c.lower() for c in df.columns]:
                    df = df.rename(columns={col: 'timestamp', col.lower(): 'timestamp'})
                    break
            
            if 'timestamp' not in df.columns and 'index' in df.columns:
                df = df.rename(columns={'index': 'timestamp'})
            
            # Limitar ao número de dias solicitado
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
        """Obtém nome da empresa pelo ticker."""
        symbol = symbol.upper()
        
        if symbol in self.COMPANY_NAMES:
            return self.COMPANY_NAMES[symbol]
        
        # Tentar obter do yfinance
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info.get('longName', info.get('shortName', symbol))
        except:
            return symbol
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Obtém preço atual."""
        df = self.get_stock_data(symbol, days=5)
        if df is not None and len(df) > 0:
            return float(df['close'].iloc[-1])
        return None
    
    def get_price_change(self, symbol: str) -> Optional[dict]:
        """Obtém variação de preço."""
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

