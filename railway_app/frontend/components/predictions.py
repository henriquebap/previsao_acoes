"""
Predictions Component - Exibicao de previsoes LSTM
"""
import streamlit as st


def render_prediction_card(prediction: dict):
    """
    Renderiza card de previsao.
    
    Args:
        prediction: Dict com current_price, predicted_price, change_percent, direction, etc.
    """
    symbol = prediction.get('symbol', '')
    current = prediction.get('current_price', 0)
    predicted = prediction.get('predicted_price', 0)
    change = prediction.get('change_percent', 0)
    direction = prediction.get('direction', '')
    confidence = prediction.get('confidence', 'Moderada')
    model_type = prediction.get('model_type', 'LSTM')
    indicators = prediction.get('indicators', {})
    
    # Cor baseada na direcao
    if change > 0:
        color = '#00ff88'
        bg_gradient = 'linear-gradient(135deg, #0a3d0c 0%, #1a4a1c 100%)'
        arrow = '↗'
    else:
        color = '#ff4757'
        bg_gradient = 'linear-gradient(135deg, #3d0a0a 0%, #4a1a1a 100%)'
        arrow = '↘'
    
    # Card principal
    st.markdown(f"""
    <div style="
        background: {bg_gradient};
        padding: 1.5rem;
        border-radius: 1rem;
        border: 1px solid {color}33;
        margin-bottom: 1rem;
    ">
        <h3 style="margin: 0; color: white;">{symbol}</h3>
        <p style="color: #888; margin: 0.5rem 0;">Previsao para amanha</p>
        
        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 1rem;">
            <div>
                <p style="color: #888; margin: 0; font-size: 0.8rem;">Atual</p>
                <p style="color: white; margin: 0; font-size: 1.5rem; font-weight: bold;">${current:.2f}</p>
            </div>
            <div style="font-size: 2rem;">{arrow}</div>
            <div>
                <p style="color: #888; margin: 0; font-size: 0.8rem;">Previsto</p>
                <p style="color: {color}; margin: 0; font-size: 1.5rem; font-weight: bold;">${predicted:.2f}</p>
            </div>
        </div>
        
        <div style="
            background: rgba(0,0,0,0.3);
            padding: 0.75rem;
            border-radius: 0.5rem;
            margin-top: 1rem;
            text-align: center;
        ">
            <span style="color: {color}; font-size: 1.5rem; font-weight: bold;">{change:+.2f}%</span>
            <p style="color: #888; margin: 0.25rem 0 0 0; font-size: 0.9rem;">{direction}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Metricas adicionais
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="
            background: rgba(255,255,255,0.05);
            padding: 1rem;
            border-radius: 0.5rem;
            text-align: center;
        ">
            <p style="color: #888; margin: 0; font-size: 0.8rem;">Confianca</p>
            <p style="color: white; margin: 0.25rem 0 0 0; font-weight: bold;">{confidence}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="
            background: rgba(255,255,255,0.05);
            padding: 1rem;
            border-radius: 0.5rem;
            text-align: center;
        ">
            <p style="color: #888; margin: 0; font-size: 0.8rem;">Modelo</p>
            <p style="color: white; margin: 0.25rem 0 0 0; font-weight: bold;">{model_type}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Indicadores tecnicos
    if indicators:
        st.markdown("### 📊 Indicadores")
        
        trend = indicators.get('trend', 'bullish')
        trend_icon = '📈' if trend == 'bullish' else '📉'
        trend_text = 'Alta' if trend == 'bullish' else 'Baixa'
        
        ind_cols = st.columns(3)
        
        with ind_cols[0]:
            st.metric("MA 7", f"${indicators.get('ma_7', 0):.2f}")
        
        with ind_cols[1]:
            st.metric("MA 30", f"${indicators.get('ma_30', 0):.2f}")
        
        with ind_cols[2]:
            st.metric("Tendencia", f"{trend_icon} {trend_text}")


def render_history_table(history: list):
    """
    Renderiza tabela de historico de previsoes.
    """
    if not history:
        st.info("Nenhuma previsao no historico")
        return
    
    st.markdown("### 📜 Historico de Previsoes")
    
    for item in history[:10]:
        symbol = item.get('symbol', '')
        predicted = item.get('predicted_price', 0)
        current = item.get('current_price', 0)
        timestamp = item.get('timestamp', '')
        
        change = ((predicted - current) / current) * 100 if current else 0
        color = '#00ff88' if change > 0 else '#ff4757'
        
        st.markdown(f"""
        <div style="
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.5rem 1rem;
            background: rgba(255,255,255,0.05);
            border-radius: 0.5rem;
            margin-bottom: 0.5rem;
        ">
            <span style="color: white; font-weight: bold;">{symbol}</span>
            <span style="color: {color};">${predicted:.2f} ({change:+.2f}%)</span>
            <span style="color: #888; font-size: 0.8rem;">{timestamp[:10]}</span>
        </div>
        """, unsafe_allow_html=True)

