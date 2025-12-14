# Correção do Problema de Recursão Infinita

## 🐛 Problema Identificado

A aplicação Streamlit estava travando/demorando muito ao selecionar ações laterais, causando erro:
```
RecursionError: maximum recursion depth exceeded while calling a Python object
```

### Causa Raiz

O problema estava em um **loop infinito de re-renderização** causado por:

1. **`st.text_input` com `value` dinâmico** (`sidebar.py` linha 56-61):
   ```python
   search_input = st.text_input(
       "Ticker ou Nome",
       value=st.session_state.get('selected_symbol', ''),  # ❌ PROBLEMA!
       ...
   )
   ```

2. **Atualização do session_state dentro do callback** (linhas 64-67):
   ```python
   if search_input:
       selected_symbol = resolve_symbol(search_input)
       if selected_symbol != st.session_state.get('selected_symbol'):
           st.session_state['selected_symbol'] = selected_symbol  # ❌ Causa loop!
   ```

3. **`st.rerun()` após atualizar estado** (linhas 121-124):
   ```python
   if st.button(ticker, ...):
       st.session_state['selected_symbol'] = ticker
       st.rerun()  # ✅ Necessário, mas causava loop com o value acima
   ```

### O que Acontecia (Loop Infinito)

```
1. Usuário clica no botão "AAPL"
2. session_state['selected_symbol'] = "AAPL"
3. st.rerun() é chamado
4. Na nova renderização, text_input recebe value="AAPL"
5. text_input muda, dispara callback
6. Atualiza session_state['selected_symbol'] novamente
7. Volta para o passo 3... (LOOP INFINITO!)
```

## ✅ Solução Implementada

### 1. Remover `value` do `text_input`
- **Antes**: `value=st.session_state.get('selected_symbol', '')`
- **Depois**: Sem `value`, apenas `key="search_input_field"`

### 2. Adicionar Flag de Controle
```python
if 'force_update_input' not in st.session_state:
    st.session_state['force_update_input'] = False

# Quando botão é clicado:
if st.button(ticker, ...):
    st.session_state['selected_symbol'] = ticker
    st.session_state['force_update_input'] = True  # ✅ Sinaliza update
    st.rerun()
```

### 3. Sincronizar Input Apenas Quando Necessário
```python
if st.session_state.get('force_update_input', False):
    st.session_state['search_input_field'] = st.session_state['selected_symbol']
    st.session_state['force_update_input'] = False
```

### 4. Prevenir Atualizações Desnecessárias
```python
if search_input:
    resolved = resolve_symbol(search_input)
    # ✅ Só atualiza se realmente mudou
    if resolved != st.session_state.get('selected_symbol'):
        st.session_state['selected_symbol'] = resolved
```

## 📊 Fluxo Correto Agora

```
1. Usuário clica no botão "AAPL"
2. session_state['selected_symbol'] = "AAPL"
3. session_state['force_update_input'] = True
4. st.rerun()
5. Na nova renderização:
   - force_update_input = True
   - Atualiza search_input_field = "AAPL"
   - force_update_input = False
6. FIM - Não cria mais loops!
```

## 🚀 Como Testar

1. **Reinicie a aplicação**:
   ```bash
   docker-compose restart frontend
   ```

2. **Teste os cenários que falhavam**:
   - ✅ Clicar em qualquer botão de ação popular (ex: AAPL, GOOGL)
   - ✅ Digitar diretamente no campo de busca
   - ✅ Alternar entre diferentes ações rapidamente
   - ✅ Usar o modo de comparação

3. **Verifique os logs** - Não deve mais aparecer:
   ```
   RecursionError: maximum recursion depth exceeded
   ```

## 📝 Arquivos Modificados

- `railway_app/frontend/components/sidebar.py` (linhas 45-77, 116-124)
- `railway_app/frontend/app.py` (linhas 568-575)

## 💡 Lições Aprendidas

### ❌ Antipadrões em Streamlit

1. **Não use `value` com `session_state` em widgets**:
   ```python
   # ❌ MAU - Cria loop
   st.text_input("Label", value=st.session_state.get('my_key', ''))
   
   # ✅ BOM - Use apenas key
   st.text_input("Label", key='my_key')
   ```

2. **Evite atualizar `session_state` do próprio widget**:
   ```python
   # ❌ MAU - Loop infinito
   search = st.text_input("Search", value=st.session_state.search)
   if search:
       st.session_state.search = search  # Cria loop!
   
   # ✅ BOM - Use key nativo
   search = st.text_input("Search", key='search')
   ```

3. **`st.rerun()` com cuidado**:
   - Use apenas quando necessário
   - Garanta que não há loops de atualização
   - Adicione flags de controle se precisar sincronizar estados

### ✅ Boas Práticas

1. **Use `key` para gerenciar estado de widgets**
2. **Separe estado de apresentação de estado de negócio**
3. **Adicione flags de controle para sincronização**
4. **Verifique mudanças antes de atualizar** (`if old != new`)

## 🔍 Monitoramento

Para evitar problemas futuros, monitore:

1. **Tempo de resposta da página**
2. **Logs de erro** (buscar por "RecursionError")
3. **Uso de CPU/Memória** (picos podem indicar loops)

## 📚 Referências

- [Streamlit Session State](https://docs.streamlit.io/library/api-reference/session-state)
- [Streamlit Caching and State](https://docs.streamlit.io/library/advanced-features/caching)
- [Common Pitfalls](https://docs.streamlit.io/library/advanced-features/app-design#common-pitfalls)

---

**Status**: ✅ Corrigido  
**Data**: 14/12/2024  
**Versão**: 1.0
