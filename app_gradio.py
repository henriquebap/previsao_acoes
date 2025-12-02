"""
Gradio interface for Stock Price Prediction API.
Deploy this on HuggingFace Spaces for an interactive demo.
"""
import gradio as gr
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

# API URL - configure this to your Railway deployment
API_URL = os.getenv("API_URL", "http://localhost:8000")


def predict_stock(symbol, days_ahead):
    """Make prediction via API."""
    try:
        response = requests.post(
            f"{API_URL}/api/v1/predict",
            json={"symbol": symbol.upper(), "days_ahead": int(days_ahead)},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            current = data['current_price']
            predicted = data['predicted_price']
            change = ((predicted - current) / current * 100)
            
            # Determine color based on change
            color = "green" if change > 0 else "red"
            arrow = "📈" if change > 0 else "📉"
            
            return f"""
## {arrow} Previsão para {symbol}

**Preço Atual:** ${current:.2f}

**Preço Previsto:** ${predicted:.2f}

**Data da Previsão:** {data['prediction_date']}

**Variação Esperada:** <span style="color: {color}; font-weight: bold;">{change:.2f}%</span>

---
*Última atualização: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
            """
        else:
            error_detail = response.json().get('detail', 'Erro desconhecido')
            return f"❌ **Erro:** {error_detail}\n\n*Certifique-se de que o modelo foi treinado para este símbolo.*"
    
    except requests.exceptions.Timeout:
        return "⏱️ **Timeout:** A API demorou muito para responder. Tente novamente."
    except requests.exceptions.ConnectionError:
        return "🔌 **Erro de Conexão:** Não foi possível conectar à API. Verifique se ela está online."
    except Exception as e:
        return f"⚠️ **Erro Inesperado:** {str(e)}"


def get_historical_data(symbol, days):
    """Get historical data via API and create chart."""
    try:
        response = requests.get(
            f"{API_URL}/api/v1/stocks/{symbol.upper()}/historical",
            params={"limit": int(days)},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            
            if not data['data']:
                return None, "📭 Nenhum dado encontrado para este símbolo."
            
            df = pd.DataFrame(data['data'])
            
            # Create candlestick chart
            fig = go.Figure(data=[go.Candlestick(
                x=df['timestamp'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name=symbol
            )])
            
            fig.update_layout(
                title=f'Histórico de Preços - {symbol.upper()}',
                xaxis_title='Data',
                yaxis_title='Preço (USD)',
                hovermode='x unified',
                template='plotly_white',
                height=500
            )
            
            # Statistics
            stats = f"""
## 📊 Estatísticas ({days} dias)

**Preço Atual:** ${df['close'].iloc[-1]:.2f}

**Máximo:** ${df['high'].max():.2f}

**Mínimo:** ${df['low'].min():.2f}

**Média:** ${df['close'].mean():.2f}

**Volatilidade:** ${df['close'].std():.2f}
            """
            
            return fig, stats
        else:
            return None, f"❌ Erro ao buscar dados: {response.json().get('detail', 'Erro desconhecido')}"
    
    except Exception as e:
        return None, f"⚠️ Erro: {str(e)}"


def get_available_stocks():
    """Get list of available stocks."""
    try:
        response = requests.get(f"{API_URL}/api/v1/stocks/available", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if not data['stocks']:
                return "📭 Nenhum modelo treinado disponível ainda."
            
            stocks_list = []
            for stock in data['stocks']:
                metrics = stock.get('metrics', {})
                rmse = metrics.get('rmse', 'N/A')
                stocks_list.append(f"- **{stock['symbol']}** (RMSE: {rmse:.4f if isinstance(rmse, float) else rmse})")
            
            return f"""
## 📈 Ações Disponíveis

{chr(10).join(stocks_list)}

*Total: {data['count']} ações*
            """
        else:
            return "❌ Erro ao buscar ações disponíveis."
    
    except Exception as e:
        return f"⚠️ Erro: {str(e)}"


# Gradio Interface
with gr.Blocks(
    title="Stock Price Predictor - LSTM",
    theme=gr.themes.Soft(),
    css=".gradio-container {max-width: 1200px; margin: auto;}"
) as demo:
    
    gr.Markdown("""
    # 📈 Stock Price Predictor - LSTM Neural Network
    
    Sistema de previsão de preços de ações usando redes neurais LSTM (Long Short-Term Memory).
    
    **Desenvolvido como parte do Tech Challenge Fase 4 - FIAP Pós-Tech MLET**
    """)
    
    with gr.Tab("🔮 Previsão"):
        gr.Markdown("### Faça uma previsão de preço de ação")
        
        with gr.Row():
            with gr.Column(scale=1):
                symbol_input = gr.Textbox(
                    label="Símbolo da Ação",
                    placeholder="Ex: AAPL, GOOGL, MSFT, AMZN",
                    value="AAPL",
                    info="Digite o ticker da ação (Yahoo Finance)"
                )
                days_input = gr.Slider(
                    minimum=1,
                    maximum=7,
                    step=1,
                    value=1,
                    label="Dias à frente",
                    info="Número de dias para prever no futuro"
                )
                predict_btn = gr.Button("🚀 Fazer Previsão", variant="primary", size="lg")
                
                gr.Markdown("""
                #### 💡 Dica
                Use símbolos como:
                - **AAPL** - Apple
                - **GOOGL** - Google
                - **MSFT** - Microsoft
                - **TSLA** - Tesla
                """)
            
            with gr.Column(scale=2):
                prediction_output = gr.Markdown(
                    "👈 Selecione uma ação e clique em 'Fazer Previsão'"
                )
        
        predict_btn.click(
            predict_stock,
            inputs=[symbol_input, days_input],
            outputs=prediction_output
        )
    
    with gr.Tab("📊 Histórico"):
        gr.Markdown("### Visualize dados históricos")
        
        with gr.Row():
            with gr.Column(scale=1):
                symbol_hist = gr.Textbox(
                    label="Símbolo da Ação",
                    placeholder="Ex: AAPL",
                    value="AAPL"
                )
                days_hist = gr.Slider(
                    minimum=30,
                    maximum=365,
                    step=30,
                    value=180,
                    label="Período (dias)"
                )
                hist_btn = gr.Button("📈 Carregar Dados", variant="primary", size="lg")
            
            with gr.Column(scale=2):
                stats_output = gr.Markdown()
        
        chart_output = gr.Plot(label="Gráfico de Candlestick")
        
        hist_btn.click(
            get_historical_data,
            inputs=[symbol_hist, days_hist],
            outputs=[chart_output, stats_output]
        )
    
    with gr.Tab("📋 Modelos Disponíveis"):
        gr.Markdown("### Ações com modelos treinados")
        
        stocks_output = gr.Markdown()
        refresh_btn = gr.Button("🔄 Atualizar Lista", variant="secondary")
        
        refresh_btn.click(
            get_available_stocks,
            outputs=stocks_output
        )
        
        # Load on startup
        demo.load(get_available_stocks, outputs=stocks_output)
    
    with gr.Tab("ℹ️ Sobre"):
        gr.Markdown("""
        ## Sobre o Projeto
        
        Este sistema utiliza **LSTM (Long Short-Term Memory)**, um tipo de rede neural recorrente,
        para prever preços de ações com base em dados históricos.
        
        ### 🎯 Características
        
        - **Modelo:** LSTM com múltiplas camadas
        - **Features:** Preços OHLC, volumes, médias móveis, volatilidade
        - **Métricas:** RMSE, MAE, MAPE, R²
        - **Dados:** Yahoo Finance (yfinance)
        
        ### 🏗️ Arquitetura
        
        1. **Coleta de Dados:** Yahoo Finance API
        2. **Feature Engineering:** Indicadores técnicos
        3. **Modelo:** PyTorch LSTM (60 dias de sequência)
        4. **API:** FastAPI com endpoints RESTful
        5. **Deploy:** Railway (API) + HuggingFace Spaces (UI)
        
        ### ⚠️ Disclaimer
        
        Este é um **projeto educacional** desenvolvido para o Tech Challenge Fase 4 da FIAP.
        
        **NÃO USE para decisões reais de investimento!** O mercado de ações é imprevisível
        e envolve riscos. Sempre consulte um profissional financeiro qualificado.
        
        ### 🔗 Links
        
        - [📖 Documentação da API](https://your-api.railway.app/docs)
        - [💻 Código Fonte (GitHub)](https://github.com/your-username/previsao_acoes)
        - [🎓 FIAP Pós-Tech MLET](https://www.fiap.com.br/graduacao/tecnologo/pos-tech-machine-learning-engineering/)
        
        ### 👨‍💻 Desenvolvido por
        
        **Seu Nome** - Tech Challenge Fase 4
        
        ---
        
        *Última atualização: 2024*
        """)

# Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )

