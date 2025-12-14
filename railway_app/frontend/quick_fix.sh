#!/bin/bash

# ============================================================================
# Quick Fix Script - Correção de Recursão Infinita
# ============================================================================

set -e  # Sair se houver erro

echo "🔧 Stock Predictor - Quick Fix Script"
echo "===================================="
echo ""

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Função para print colorido
print_step() {
    echo -e "${BLUE}▶ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Verificar se estamos na pasta correta
if [ ! -f "app.py" ]; then
    print_error "Execute este script na pasta railway_app/frontend/"
    exit 1
fi

print_success "Pasta correta detectada"
echo ""

# ============================================================================
# 1. Verificar mudanças
# ============================================================================
print_step "1. Verificando mudanças nos arquivos..."
echo ""

if git diff --quiet components/sidebar.py app.py; then
    print_warning "Nenhuma mudança detectada. Talvez já esteja aplicado?"
else
    print_success "Mudanças detectadas em:"
    git diff --name-only
fi

echo ""

# ============================================================================
# 2. Parar containers
# ============================================================================
print_step "2. Parando containers existentes..."
cd ..
docker-compose down
print_success "Containers parados"
echo ""

# ============================================================================
# 3. Rebuild (opcional)
# ============================================================================
read -p "$(echo -e ${YELLOW}Fazer rebuild completo? [s/N]: ${NC})" rebuild
if [[ $rebuild =~ ^[Ss]$ ]]; then
    print_step "3. Fazendo rebuild do frontend..."
    docker-compose build --no-cache frontend
    print_success "Rebuild concluído"
else
    print_step "3. Pulando rebuild (usando cache)..."
fi
echo ""

# ============================================================================
# 4. Iniciar containers
# ============================================================================
print_step "4. Iniciando containers..."
docker-compose up -d
print_success "Containers iniciados"
echo ""

# ============================================================================
# 5. Verificar saúde
# ============================================================================
print_step "5. Verificando saúde dos containers..."
sleep 3

if docker ps | grep -q "railway_app-frontend"; then
    print_success "Frontend está rodando"
else
    print_error "Frontend não está rodando!"
    exit 1
fi

if docker ps | grep -q "railway_app-backend"; then
    print_success "Backend está rodando"
else
    print_warning "Backend não está rodando (pode ser necessário)"
fi
echo ""

# ============================================================================
# 6. Mostrar logs
# ============================================================================
print_step "6. Últimos logs do frontend:"
echo ""
docker-compose logs --tail=20 frontend
echo ""

# ============================================================================
# 7. Verificar erros críticos
# ============================================================================
print_step "7. Verificando por RecursionError..."
if docker-compose logs frontend | grep -q "RecursionError"; then
    print_error "AINDA HÁ RecursionError nos logs!"
    echo "Execute: docker-compose logs frontend | grep -A 5 RecursionError"
    exit 1
else
    print_success "Nenhum RecursionError encontrado"
fi
echo ""

# ============================================================================
# 8. Informações de acesso
# ============================================================================
print_step "8. Informações de Acesso:"
echo ""
echo -e "${GREEN}📱 Frontend:${NC} http://localhost:8501"
echo -e "${GREEN}🔧 Backend:${NC}  http://localhost:8000"
echo -e "${GREEN}📊 Docs API:${NC} http://localhost:8000/docs"
echo ""

# ============================================================================
# 9. Próximos passos
# ============================================================================
print_step "9. Próximos Passos:"
echo ""
echo "1. Abra o navegador: http://localhost:8501"
echo "2. Teste clicar nas ações populares (AAPL, GOOGL, etc.)"
echo "3. Verifique se não trava mais"
echo "4. Monitore os logs: docker-compose logs -f frontend"
echo ""
echo "Para testar em modo interativo:"
echo "  docker-compose logs -f frontend | grep -i error"
echo ""

# ============================================================================
# 10. Menu de opções
# ============================================================================
echo -e "${BLUE}═══════════════════════════════════════════${NC}"
echo -e "${YELLOW}Escolha uma opção:${NC}"
echo ""
echo "1) 📊 Ver logs em tempo real"
echo "2) 🔍 Buscar por erros"
echo "3) 🔄 Reiniciar apenas frontend"
echo "4) 🧪 Executar testes"
echo "5) 🚀 Preparar para deploy (git)"
echo "6) ❌ Sair"
echo ""
read -p "Opção: " option

case $option in
    1)
        print_step "Mostrando logs em tempo real (Ctrl+C para sair)..."
        docker-compose logs -f frontend
        ;;
    2)
        print_step "Buscando por erros..."
        echo ""
        docker-compose logs frontend | grep -i -E "(error|exception|recursion)" --color=always || echo "Nenhum erro encontrado!"
        ;;
    3)
        print_step "Reiniciando frontend..."
        docker-compose restart frontend
        sleep 3
        print_success "Frontend reiniciado"
        docker-compose logs --tail=10 frontend
        ;;
    4)
        print_step "Executando testes..."
        echo "Abrindo navegador para testes manuais..."
        open "http://localhost:8501" 2>/dev/null || xdg-open "http://localhost:8501" 2>/dev/null || echo "Abra manualmente: http://localhost:8501"
        ;;
    5)
        print_step "Preparando para deploy..."
        cd frontend
        echo ""
        echo "Arquivos modificados:"
        git status --short
        echo ""
        read -p "$(echo -e ${YELLOW}Deseja fazer commit? [s/N]: ${NC})" do_commit
        if [[ $do_commit =~ ^[Ss]$ ]]; then
            git add components/sidebar.py app.py FIX_RECURSION.md TEST_INSTRUCTIONS.md quick_fix.sh
            git commit -m "fix: corrige recursão infinita no Streamlit

- Remove value dinâmico do st.text_input
- Adiciona flag force_update_input para controle
- Previne loop infinito de reruns
- Adiciona documentação da correção"
            print_success "Commit criado!"
            echo ""
            read -p "$(echo -e ${YELLOW}Fazer push para deploy? [s/N]: ${NC})" do_push
            if [[ $do_push =~ ^[Ss]$ ]]; then
                git push
                print_success "Push realizado! Railway vai fazer deploy automaticamente."
            fi
        fi
        ;;
    6)
        print_success "Finalizado!"
        ;;
    *)
        print_warning "Opção inválida"
        ;;
esac

echo ""
print_success "Script concluído!"
echo ""
echo -e "${GREEN}═══════════════════════════════════════════${NC}"
echo -e "${GREEN}✨ Correção aplicada com sucesso!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════${NC}"
