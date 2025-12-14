# ⚡ Guia Rápido - Correção de Recursão

## 🚀 Aplicar Correção (2 minutos)

```bash
cd railway_app/frontend
./quick_fix.sh
```

Siga as instruções na tela.

---

## 🧪 Teste Rápido (1 minuto)

1. Abra: http://localhost:8501
2. Clique em **AAPL** (ou qualquer ação)
3. ✅ Deve carregar em < 2 segundos
4. ❌ Não deve travar ou mostrar erros

---

## 📊 Verificar Logs

```bash
# Ver logs em tempo real
docker-compose logs -f frontend

# Buscar erros
docker-compose logs frontend | grep -i error

# ✅ Logs saudáveis:
# "You can now view your Streamlit app"

# ❌ Logs problemáticos (não devem aparecer):
# "RecursionError"
# "maximum recursion depth exceeded"
```

---

## 🌐 Deploy Produção

```bash
# 1. Commit
git add railway_app/frontend
git commit -m "fix: corrige recursão infinita no Streamlit"

# 2. Push
git push origin main

# 3. Aguardar deploy no Railway (~3-5 min)

# 4. Testar
# Abrir: https://stock-pred.up.railway.app
```

---

## 🔧 Problemas?

### Ainda trava?
```bash
docker-compose down -v
docker-compose up --build
```

### Limpar tudo?
```bash
docker-compose down -v
docker system prune -a --volumes -f
docker-compose up --build
```

### Verificar container?
```bash
docker ps
docker-compose logs frontend --tail=50
```

---

## 📁 Arquivos Modificados

- ✅ `components/sidebar.py` (linhas 45-77, 116-124)
- ✅ `app.py` (linhas 573-575)

## 📚 Documentação Completa

- **Detalhes técnicos**: `FIX_RECURSION.md`
- **Testes completos**: `TEST_INSTRUCTIONS.md`
- **Resumo geral**: `/CORREÇÃO_RECURSAO.md`

---

## ✅ Checklist

- [ ] Script executado
- [ ] Teste local OK
- [ ] Logs limpos
- [ ] Deploy realizado
- [ ] Produção testada

---

**Dúvidas?** Leia a documentação completa.
