"""
Predictions Component - Exibicao de previsoes LSTM
Usando componentes nativos do Streamlit para melhor compatibilidade
"""
import streamlit as st


def render_prediction_card(prediction: dict):
    """
    Renderiza card de previsao usando componentes nativos.
    """
    symbol = prediction.get('symbol', '')
    current = prediction.get('current_price', 0)
    predicted = prediction.get('predicted_price', 0)
    change = prediction.get('change_percent', 0)
    direction = prediction.get('direction', '')
    confidence = prediction.get('confidence', 'Moderada')
    model_type = prediction.get('model_type', 'LSTM')
    indicators = prediction.get('indicators', {})
    
    # Determinar cor e emoji
    if change > 0:
        delta_color = "normal"  # verde
        arrow = "📈"
    else:
        delta_color = "inverse"  # vermelho
        arrow = "📉"
    
    # Container com estilo
    st.markdown(f"### {arrow} Previsão {symbol}")
    
    # Metricas principais em colunas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="💰 Preço Atual",
            value=f"${current:.2f}"
        )
    
    with col2:
        st.metric(
            label="🔮 Previsão",
            value=f"${predicted:.2f}",
            delta=f"{change:+.2f}%",
            delta_color=delta_color
        )
    
    with col3:
        st.metric(
            label="📊 Direção",
            value=direction.split(' ')[-1] if ' ' in direction else direction
        )
    
    # Info do modelo
    st.markdown("---")
    
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.info(f"🧠 **Modelo**: {model_type}")
    
    with info_col2:
        st.info(f"🎯 **Confiança**: {confidence}")
    
    # Indicadores tecnicos
    if indicators:
        st.markdown("#### 📈 Indicadores Técnicos")
        
        ind_col1, ind_col2, ind_col3 = st.columns(3)
        
        with ind_col1:
            ma7 = indicators.get('ma_7', 0)
            st.metric("MA 7 dias", f"${ma7:.2f}")
        
        with ind_col2:
            ma30 = indicators.get('ma_30', 0)
            st.metric("MA 30 dias", f"${ma30:.2f}")
        
        with ind_col3:
            trend = indicators.get('trend', 'bullish')
            trend_text = "📈 Alta" if trend == 'bullish' else "📉 Baixa"
            st.metric("Tendência", trend_text)
    
    # Disclaimer
    st.caption("⚠️ Previsão educacional. NÃO use para investimentos reais!")


def render_history_table(history: list):
    """Renderiza tabela de historico de previsoes."""
    if not history:
        st.info("Nenhuma previsao no historico")
        return
    
    st.markdown("### 📜 Histórico de Previsões")
    
    for item in history[:10]:
        symbol = item.get('symbol', '')
        predicted = item.get('predicted_price', 0)
        current = item.get('current_price', 0)
        timestamp = item.get('timestamp', '')
        
        change = ((predicted - current) / current) * 100 if current else 0
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            st.write(f"**{symbol}**")
        with col2:
            delta_color = "normal" if change > 0 else "inverse"
            st.metric("", f"${predicted:.2f}", f"{change:+.2f}%", delta_color=delta_color, label_visibility="collapsed")
        with col3:
            st.caption(timestamp[:10] if timestamp else "")
