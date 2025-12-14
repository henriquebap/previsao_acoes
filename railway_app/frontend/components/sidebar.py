"""
Sidebar Component - Lista de acoes e configuracoes
"""
import streamlit as st
from typing import Tuple, List


POPULAR_TICKERS = {
    "🇺🇸 Tech US": ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "NVDA", "TSLA", "NFLX"],
    "💰 Finance US": ["JPM", "BAC", "V", "MA", "GS", "BRK-B"],
    "🛒 Consumer US": ["WMT", "KO", "MCD", "SBUX", "NKE", "DIS"],
    "💊 Healthcare US": ["JNJ", "PFE", "UNH", "MRNA"],
    "🇧🇷 Brasil B3": ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "ABEV3.SA", "WEGE3.SA"]
}

COMPANY_MAP = {
    "apple": "AAPL", "google": "GOOGL", "microsoft": "MSFT",
    "amazon": "AMZN", "meta": "META", "nvidia": "NVDA",
    "tesla": "TSLA", "netflix": "NFLX", "petrobras": "PETR4.SA",
    "vale": "VALE3.SA", "itau": "ITUB4.SA", "bradesco": "BBDC4.SA"
}


def resolve_symbol(query: str) -> str:
    """Converte nome para ticker."""
    query = query.strip()
    if not query:
        return ""
    
    # Se ja e um ticker em maiusculas
    if query.upper() == query and len(query) <= 10:
        return query.upper()
    
    key = query.lower()
    if key in COMPANY_MAP:
        return COMPANY_MAP[key]
    
    for company, ticker in COMPANY_MAP.items():
        if key in company:
            return ticker
    
    return query.upper()


def render_sidebar() -> Tuple[str, int, bool, List[str], bool]:
    """Renderiza sidebar e retorna selecoes + trigger de load."""
    
    # Inicializar session_state
    if 'selected_symbol' not in st.session_state:
        st.session_state['selected_symbol'] = ''
    if 'loaded_symbol' not in st.session_state:
        st.session_state['loaded_symbol'] = ''
    if 'search_input_field' not in st.session_state:
        st.session_state['search_input_field'] = ''
    
    with st.sidebar:
        st.markdown("## 🔍 Buscar Ação")
        
        # Input de busca
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_input = st.text_input(
                "Ticker ou Nome",
                placeholder="Ex: AAPL",
                key="search_input_field",
                label_visibility="collapsed"
            )
        
        # Resolver símbolo
        if search_input:
            selected_symbol = resolve_symbol(search_input)
        else:
            selected_symbol = st.session_state.get('selected_symbol', '')
        
        # Botão Carregar - TRIGGER EXPLÍCITO
        with col2:
            load_button = st.button(
                "🚀",
                key="load_button",
                help="Carregar dados",
                use_container_width=True,
                disabled=not selected_symbol
            )
        
        # Status da seleção
        if selected_symbol:
            if selected_symbol == st.session_state.get('loaded_symbol'):
                st.info(f"📊 **{selected_symbol}** (carregado)")
            else:
                st.warning(f"⚠️ **{selected_symbol}** (não carregado)\n\nClique em 🚀 para carregar")
        
        # Atualizar estado quando carregar
        should_load = False
        if load_button and selected_symbol:
            st.session_state['selected_symbol'] = selected_symbol
            st.session_state['loaded_symbol'] = selected_symbol
            should_load = True
        
        st.markdown("---")
        
        # Periodo
        st.markdown("## 📅 Período")
        days_options = {
            "1 Mês": 30,
            "3 Meses": 90,
            "6 Meses": 180,
            "1 Ano": 365,
            "2 Anos": 730
        }
        
        selected_period = st.selectbox(
            "Selecione o período",
            options=list(days_options.keys()),
            index=2
        )
        selected_days = days_options[selected_period]
        
        st.markdown("---")
        
        # Modo comparacao
        st.markdown("## 📊 Comparação")
        compare_mode = st.checkbox("Comparar ações", value=False)
        
        compare_symbols = []
        if compare_mode:
            compare_input = st.text_input(
                "Símbolos (separados por vírgula)",
                placeholder="AAPL, GOOGL, MSFT",
                key="compare_input"
            )
            if compare_input:
                compare_symbols = [resolve_symbol(s.strip()) for s in compare_input.split(",")]
                st.info(f"Comparando: {', '.join(compare_symbols)}")
        
        st.markdown("---")
        
        # Acoes populares
        st.markdown("## ⭐ Populares")
        
        for category, tickers in POPULAR_TICKERS.items():
            with st.expander(category, expanded=False):
                cols = st.columns(2)
                for i, ticker in enumerate(tickers):
                    with cols[i % 2]:
                        # Indicador se já está carregado
                        is_loaded = ticker == st.session_state.get('loaded_symbol')
                        button_label = f"✓ {ticker}" if is_loaded else ticker
                        
                        if st.button(button_label, key=f"btn_{ticker}", use_container_width=True):
                            # Selecionar E carregar imediatamente
                            st.session_state['selected_symbol'] = ticker
                            st.session_state['loaded_symbol'] = ticker
                            st.session_state['search_input_field'] = ticker
                            should_load = True
                            st.rerun()
        
        st.markdown("---")
        
        # Info
        st.markdown("""
        ### 💡 Dicas
        - Digite o nome da empresa
        - Ou use o ticker (AAPL)
        - Brasil: sufixo .SA
        """)
        
        st.markdown("""
        <div style='text-align: center; font-size: 0.8rem; color: #888; margin-top: 1rem;'>
            🎓 FIAP Pós-Tech MLET<br>
            Tech Challenge Fase 4
        </div>
        """, unsafe_allow_html=True)
    
    return selected_symbol, selected_days, compare_mode, compare_symbols, should_load
