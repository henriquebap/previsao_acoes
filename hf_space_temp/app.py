"""
Stock Price Predictor - LSTM
HuggingFace Spaces - FIAP Tech Challenge Fase 4

Features:
- Suporte a linguagem natural (Apple → AAPL)
- 50+ tickers populares BR e internacionais
- Modelo LSTM do HuggingFace Hub
"""
import gradio as gr
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from huggingface_hub import hf_hub_download
import joblib
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# MAPEAMENTO DE EMPRESAS (Linguagem Natural → Ticker)
# ============================================================================

COMPANY_TO_TICKER = {
    # Tech Giants - US
    "apple": "AAPL", "maçã": "AAPL",
    "google": "GOOGL", "alphabet": "GOOGL",
    "microsoft": "MSFT",
    "amazon": "AMZN",
    "meta": "META", "facebook": "META", "fb": "META",
    "nvidia": "NVDA",
    "tesla": "TSLA",
    "netflix": "NFLX",
    "adobe": "ADBE",
    "salesforce": "CRM",
    "intel": "INTC",
    "amd": "AMD",
    "ibm": "IBM",
    "oracle": "ORCL",
    "cisco": "CSCO",
    "paypal": "PYPL",
    "uber": "UBER",
    "airbnb": "ABNB",
    "spotify": "SPOT",
    "zoom": "ZM",
    "shopify": "SHOP",
    "twitter": "TWTR", "x": "TWTR",
    
    # Finance - US
    "jpmorgan": "JPM", "jp morgan": "JPM",
    "bank of america": "BAC", "bofa": "BAC",
    "wells fargo": "WFC",
    "goldman sachs": "GS",
    "morgan stanley": "MS",
    "visa": "V",
    "mastercard": "MA",
    "american express": "AXP", "amex": "AXP",
    "berkshire": "BRK-B", "buffett": "BRK-B",
    
    # Consumer - US
    "walmart": "WMT",
    "coca cola": "KO", "coca-cola": "KO", "coke": "KO",
    "pepsi": "PEP", "pepsico": "PEP",
    "mcdonalds": "MCD", "mcdonald's": "MCD",
    "starbucks": "SBUX",
    "nike": "NKE",
    "disney": "DIS",
    "home depot": "HD",
    "costco": "COST",
    "target": "TGT",
    
    # Healthcare - US
    "johnson & johnson": "JNJ", "j&j": "JNJ",
    "pfizer": "PFE",
    "moderna": "MRNA",
    "unitedhealth": "UNH",
    "merck": "MRK",
    "abbvie": "ABBV",
    
    # Energy - US
    "exxon": "XOM", "exxonmobil": "XOM",
    "chevron": "CVX",
    
    # BRASIL - B3
    "petrobras": "PETR4.SA", "petro": "PETR4.SA",
    "vale": "VALE3.SA",
    "itau": "ITUB4.SA", "itaú": "ITUB4.SA",
    "bradesco": "BBDC4.SA",
    "banco do brasil": "BBAS3.SA", "bb": "BBAS3.SA",
    "ambev": "ABEV3.SA",
    "magazine luiza": "MGLU3.SA", "magalu": "MGLU3.SA",
    "weg": "WEGE3.SA",
    "b3": "B3SA3.SA",
    "nubank": "NU",
    "mercado livre": "MELI",
}

# Tickers populares para exibição
POPULAR_TICKERS = {
    "🇺🇸 Tech US": ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "NVDA", "TSLA", "NFLX"],
    "💰 Finance US": ["JPM", "BAC", "V", "MA", "GS", "BRK-B"],
    "🛒 Consumer US": ["WMT", "KO", "MCD", "SBUX", "NKE", "DIS"],
    "💊 Healthcare US": ["JNJ", "PFE", "UNH", "MRNA"],
    "🇧🇷 Brasil B3": ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "ABEV3.SA", "WEGE3.SA"],
}


def resolve_ticker(user_input: str) -> str:
    """Converte linguagem natural para ticker."""
    user_input = user_input.strip()
    
    # Se já é um ticker válido (maiúsculas, 1-5 chars)
    if user_input.upper() == user_input and 1 <= len(user_input) <= 10:
        return user_input.upper()
    
    # Procurar no mapeamento
    key = user_input.lower()
    if key in COMPANY_TO_TICKER:
        return COMPANY_TO_TICKER[key]
    
    # Procurar parcial
    for company, ticker in COMPANY_TO_TICKER.items():
        if key in company or company in key:
            return ticker
    
    # Retornar como está (pode ser ticker desconhecido)
    return user_input.upper()


# ============================================================================
# LSTM MODEL
# ============================================================================

class LSTMModel(nn.Module):
    def __init__(self, input_size=16, hidden_size=50, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
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


# ============================================================================
# DATA FUNCTIONS (com fallback robusto)
# ============================================================================

def load_stock_data(symbol: str, days: int = 400) -> pd.DataFrame:
    """Carrega dados com múltiplas tentativas."""
    import yfinance as yf
    
    end = datetime.now()
    start = end - timedelta(days=days)
    
    # Tentar múltiplas vezes
    for attempt in range(3):
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start.strftime('%Y-%m-%d'),
                end=end.strftime('%Y-%m-%d'),
                auto_adjust=True
            )
            
            if not df.empty:
                df = df.reset_index()
                df.columns = df.columns.str.lower()
                if 'date' in df.columns:
                    df = df.rename(columns={'date': 'timestamp'})
                return df
                
        except Exception as e:
            if attempt == 2:
                raise ValueError(f"Não foi possível obter dados para {symbol}: {str(e)}")
            continue
    
    raise ValueError(f"Dados não encontrados para {symbol}")


def create_features(df: pd.DataFrame) -> pd.DataFrame:
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
    
    # Limpar infinitos
    df = df.replace([np.inf, -np.inf], 0)
    
    return df.bfill().ffill()


# ============================================================================
# MODEL HUB
# ============================================================================

MODEL_REPO = "henriquebap/stock-predictor-lstm"
model_cache = {}


def load_model_from_hub(symbol: str):
    """Carrega modelo do Hub."""
    if symbol in model_cache:
        return model_cache[symbol]
    
    try:
        model_file = f"lstm_model_{symbol}.pth"
        scaler_file = f"scaler_{symbol}.pkl"
        
        try:
            model_path = hf_hub_download(repo_id=MODEL_REPO, filename=model_file)
            scaler_path = hf_hub_download(repo_id=MODEL_REPO, filename=scaler_file)
            model_type = "específico"
        except:
            model_path = hf_hub_download(repo_id=MODEL_REPO, filename="lstm_model_BASE.pth")
            scaler_path = hf_hub_download(repo_id=MODEL_REPO, filename="scaler_BASE.pkl")
            model_type = "base"
        
        checkpoint = torch.load(model_path, map_location='cpu')
        
        model = LSTMModel(
            input_size=checkpoint['input_size'],
            hidden_size=checkpoint['hidden_size'],
            num_layers=checkpoint['num_layers'],
            dropout=checkpoint.get('dropout', 0.2)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        scaler_data = joblib.load(scaler_path)
        
        model_cache[symbol] = {
            'model': model,
            'scaler': scaler_data['scaler'],
            'target_scaler': scaler_data['target_scaler'],
            'feature_columns': scaler_data['feature_columns'],
            'type': model_type
        }
        
        return model_cache[symbol]
    except Exception as e:
        return None


# ============================================================================
# PREDICTION
# ============================================================================

def predict_stock(user_input: str) -> str:
    """Faz previsão com suporte a linguagem natural."""
    
    if not user_input or not user_input.strip():
        return "❌ Digite o nome de uma empresa ou ticker (ex: Apple, AAPL, Petrobras)"
    
    # Resolver ticker
    symbol = resolve_ticker(user_input)
    original_input = user_input.strip()
    
    try:
        # Carregar dados
        df = load_stock_data(symbol)
        
        if len(df) < 70:
            return f"❌ Dados insuficientes para {symbol} (mínimo 70 dias)"
        
        current_price = float(df['close'].iloc[-1])
        
        # Preparar features
        df_feat = create_features(df)
        
        # Tentar modelo LSTM do Hub
        model_data = load_model_from_hub(symbol)
        
        if model_data:
            # Usar LSTM
            try:
                feature_cols = model_data['feature_columns']
                for col in feature_cols:
                    if col not in df_feat.columns:
                        df_feat[col] = 0
                
                features = df_feat[feature_cols].values
                features_scaled = model_data['scaler'].transform(features)
                
                X = features_scaled[-60:].reshape(1, 60, len(feature_cols))
                X_tensor = torch.FloatTensor(X)
                
                with torch.no_grad():
                    pred_scaled = model_data['model'](X_tensor).numpy()[0, 0]
                
                predicted_price = model_data['target_scaler'].inverse_transform([[pred_scaled]])[0, 0]
                model_type = f"LSTM {model_data['type'].capitalize()}"
            except:
                # Fallback
                predicted_price = current_price * (1 + float(df_feat['momentum'].iloc[-1]) / current_price * 0.5)
                model_type = "Fallback"
        else:
            # Modelo simples
            ma_7 = float(df_feat['ma_7'].iloc[-1])
            momentum = float(df_feat['momentum'].iloc[-1])
            predicted_price = current_price + momentum * 0.3
            model_type = "Técnico"
        
        # Calcular métricas
        change_pct = ((predicted_price - current_price) / current_price) * 100
        
        if change_pct > 1:
            direction = "📈 ALTA"
            emoji = "🟢"
        elif change_pct < -1:
            direction = "📉 BAIXA"
            emoji = "🔴"
        else:
            direction = "➡️ LATERAL"
            emoji = "🟡"
        
        # Indicadores
        ma_7 = float(df_feat['ma_7'].iloc[-1])
        ma_30 = float(df_feat['ma_30'].iloc[-1])
        volatility = float(df_feat['volatility_7'].iloc[-1])
        trend = "📈 Positiva" if ma_7 > ma_30 else "📉 Negativa"
        
        # Performance
        week_change = ((current_price - float(df['close'].iloc[-5])) / float(df['close'].iloc[-5])) * 100 if len(df) > 5 else 0
        month_change = ((current_price - float(df['close'].iloc[-21])) / float(df['close'].iloc[-21])) * 100 if len(df) > 21 else 0
        
        # Mostrar conversão se houve
        input_info = f"**Pesquisa**: {original_input} → **{symbol}**" if original_input.lower() != symbol.lower() else f"**Ticker**: {symbol}"
        
        return f"""
# {emoji} {direction} prevista para {symbol}

{input_info}

---

## 🤖 Modelo: {model_type}

| Métrica | Valor |
|---------|-------|
| **Preço Atual** | ${current_price:.2f} |
| **Previsão** | ${predicted_price:.2f} |
| **Variação** | {change_pct:+.2f}% |

---

## 📊 Indicadores Técnicos

| Indicador | Valor |
|-----------|-------|
| **MA 7 dias** | ${ma_7:.2f} |
| **MA 30 dias** | ${ma_30:.2f} |
| **Tendência** | {trend} |
| **Volatilidade** | ${volatility:.2f} |

---

## 📅 Performance Recente

| Período | Variação |
|---------|----------|
| **Semana** | {week_change:+.2f}% |
| **Mês** | {month_change:+.2f}% |

---

⚠️ **Disclaimer**: Previsão educacional. NÃO use para investimentos!

*🎓 Tech Challenge Fase 4 - FIAP Pós-Tech MLET*
"""
        
    except Exception as e:
        return f"""
❌ **Erro ao processar "{user_input}"**

**Possíveis causas:**
- Ticker inválido ou não encontrado
- API do Yahoo Finance temporariamente indisponível
- Empresa não listada na bolsa

**Tente:**
- Usar o ticker oficial (ex: AAPL, GOOGL, PETR4.SA)
- Verificar se a empresa está listada
- Aguardar alguns minutos e tentar novamente

**Erro técnico:** {str(e)[:100]}
"""


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

# Criar tabela de tickers populares
def create_ticker_table():
    lines = ["## 📋 Tickers Populares\n"]
    for category, tickers in POPULAR_TICKERS.items():
        lines.append(f"**{category}**: {', '.join(tickers)}")
    return "\n\n".join(lines)


with gr.Blocks(title="Stock Predictor LSTM") as demo:
    gr.Markdown("""
    # 📈 Stock Price Predictor - LSTM
    
    ### Sistema de Previsão com Deep Learning
    
    🎓 **Tech Challenge Fase 4** - FIAP Pós-Tech Machine Learning Engineering
    
    ---
    
    **💡 Dica**: Digite o nome da empresa ou o ticker!
    - `Apple` ou `AAPL`
    - `Nvidia` ou `NVDA`
    - `Petrobras` ou `PETR4.SA`
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                label="🔍 Empresa ou Ticker",
                placeholder="Ex: Apple, GOOGL, Petrobras, NVDA...",
                value="Apple"
            )
            submit_btn = gr.Button("🚀 Analisar", variant="primary")
            
            gr.Markdown(create_ticker_table())
            
        with gr.Column(scale=2):
            output = gr.Markdown(label="Resultado")
    
    gr.Markdown("""
    ---
    
    ### 🧠 Sobre o Modelo
    
    - **Arquitetura**: LSTM 2 camadas × 50 neurônios
    - **Features**: 16 indicadores técnicos
    - **Período**: 60 dias de histórico
    - **Modelos Treinados**: AAPL, GOOGL (outros usam modelo BASE)
    
    📦 **Model Hub**: [henriquebap/stock-predictor-lstm](https://huggingface.co/henriquebap/stock-predictor-lstm)
    
    ---
    
    *Dezembro 2024 | FIAP Pós-Tech MLET*
    """)
    
    submit_btn.click(fn=predict_stock, inputs=input_text, outputs=output)
    input_text.submit(fn=predict_stock, inputs=input_text, outputs=output)


if __name__ == "__main__":
    demo.launch()
