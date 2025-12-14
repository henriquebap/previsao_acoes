# 🧪 Instruções de Teste - Correção de Recursão

## 🎯 Objetivo

Validar que o problema de recursão infinita foi corrigido e a aplicação está funcionando corretamente.

## 🚀 Como Testar Localmente

### 1. Rebuild e Restart do Container

```bash
# Navegue até a pasta do projeto
cd /Users/henriquebap/Pessoal/PosTech/previsao_acoes/railway_app

# Pare os containers
docker-compose down

# Rebuild o frontend (força reconstrução)
docker-compose build --no-cache frontend

# Inicie novamente
docker-compose up -d

# Veja os logs em tempo real
docker-compose logs -f frontend
```

### 2. Acesse a Aplicação

Abra no navegador:
```
http://localhost:8501
```

### 3. Cenários de Teste

#### ✅ Teste 1: Clicar em Ação Popular
1. Na sidebar, expanda qualquer categoria (ex: "🇺🇸 Tech US")
2. Clique em qualquer botão (ex: **AAPL**)
3. **Esperado**: 
   - Página carrega rapidamente (< 2s)
   - Mostra "✅ Selecionado: AAPL" na sidebar
   - Campo de busca é preenchido com "AAPL"
   - Dados da ação aparecem no gráfico
4. **NÃO deve acontecer**:
   - Página congelar
   - Mensagem de erro no console
   - Recarregamentos infinitos

#### ✅ Teste 2: Digitar no Campo de Busca
1. Clique no campo "Ticker ou Nome"
2. Digite "GOOGL"
3. **Esperado**:
   - Mostra "✅ Selecionado: GOOGL"
   - Dados carregam automaticamente
4. Limpe o campo e digite "apple"
5. **Esperado**:
   - Resolve para "AAPL"
   - Mostra os dados da Apple

#### ✅ Teste 3: Alternar Entre Ações Rapidamente
1. Clique em **AAPL**
2. Imediatamente clique em **GOOGL**
3. Imediatamente clique em **MSFT**
4. **Esperado**:
   - Cada clique responde rápido
   - Página atualiza sem travar
   - Logs não mostram erros

#### ✅ Teste 4: Modo Comparação
1. Na sidebar, marque "Comparar ações"
2. Digite "AAPL, GOOGL, MSFT"
3. **Esperado**:
   - Mostra gráfico de comparação
   - Não há erros de recursão

#### ✅ Teste 5: Página de Monitoramento
1. Clique em "📊 Monitoramento" no menu
2. Clique em "🔄 Atualizar"
3. **Esperado**:
   - Página atualiza normalmente
   - Métricas aparecem

#### ✅ Teste 6: Fazer Previsão
1. Selecione uma ação (ex: AAPL)
2. Clique em "🚀 Fazer Previsão"
3. **Esperado**:
   - Loading aparece
   - Previsão é exibida
   - Página não trava

## 🔍 Verificar Logs

### Logs SAUDÁVEIS (esperados):
```bash
# Execute:
docker-compose logs frontend

# Deve mostrar:
Starting Container
Collecting usage statistics...
You can now view your Streamlit app in your browser.
URL: http://0.0.0.0:8501

# Ao clicar nas ações, pode aparecer (é normal):
Session with id xxx is already connected! Connecting to a new session.
```

### Logs PROBLEMÁTICOS (NÃO devem aparecer):
```bash
❌ RecursionError: maximum recursion depth exceeded
❌ Exception in thread ScriptRunner.scriptThread
❌ [Previous line repeated 975 more times]
```

## 📊 Métricas de Performance

### Antes da Correção:
- ❌ Clique na ação: 10-30s (ou infinito)
- ❌ CPU: 100% constante
- ❌ Logs: RecursionError após cada clique

### Depois da Correção:
- ✅ Clique na ação: < 2s
- ✅ CPU: 5-20% normal
- ✅ Logs: Limpos, sem erros

## 🌐 Testar em Produção (Railway)

### 1. Deploy Manual
```bash
# Navegue até a pasta frontend
cd railway_app/frontend

# Commit as mudanças
git add .
git commit -m "fix: corrige recursão infinita no Streamlit"

# Push para trigger deploy no Railway
git push origin main
```

### 2. Aguarde Deploy
- Acesse o painel do Railway
- Aguarde o deploy completar (~3-5 min)
- Acesse a URL de produção: https://stock-pred.up.railway.app

### 3. Execute Todos os Testes Acima
- Repita os 6 cenários de teste
- Verifique logs no Railway Dashboard

## 🐛 Se Ainda Houver Problemas

### Debug Adicional

1. **Verifique versões**:
```bash
# Entre no container
docker exec -it railway_app-frontend-1 bash

# Verifique versão do Streamlit
pip show streamlit

# Deve ser >= 1.29.0
```

2. **Limpe Cache do Streamlit**:
```bash
# Dentro do container
rm -rf /root/.streamlit/cache
```

3. **Force rebuild completo**:
```bash
docker-compose down -v
docker system prune -a --volumes -f
docker-compose up --build
```

4. **Verifique estado do navegador**:
- Abra DevTools (F12)
- Console → Verifique erros JavaScript
- Network → Veja se há requests infinitos

### Logs Detalhados

```bash
# Ative debug mode
# Em app.py, adicione no topo:
import logging
logging.basicConfig(level=logging.DEBUG)

# Ou configure no Streamlit:
# .streamlit/config.toml
[logger]
level = "debug"
```

## 📋 Checklist Final

Antes de considerar concluído:

- [ ] ✅ Teste 1: Clicar em ações populares funciona
- [ ] ✅ Teste 2: Busca por texto funciona
- [ ] ✅ Teste 3: Alternar rapidamente funciona
- [ ] ✅ Teste 4: Modo comparação funciona
- [ ] ✅ Teste 5: Página de monitoramento funciona
- [ ] ✅ Teste 6: Fazer previsão funciona
- [ ] ✅ Logs sem RecursionError
- [ ] ✅ CPU/Memória em níveis normais
- [ ] ✅ Testado em produção (Railway)

## 📞 Suporte

Se persistir algum problema:

1. **Documente**:
   - Screenshot do erro
   - Logs completos (`docker-compose logs frontend > logs.txt`)
   - Passos para reproduzir

2. **Verifique**:
   - Versões (Python, Streamlit, Docker)
   - Configurações de ambiente
   - Estado do banco de dados

3. **Tente**:
   - Limpar cache do navegador
   - Usar aba anônima
   - Testar em outro navegador
   - Reiniciar containers completamente

---

**Data**: 14/12/2024  
**Versão**: 1.0  
**Status**: ✅ Pronto para testar
