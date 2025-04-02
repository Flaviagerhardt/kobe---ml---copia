#!/bin/bash

# Função para exibir ajuda
show_help() {
    echo "Script para facilitar a execução do Projeto Kobe ML"
    echo ""
    echo "Uso: ./run.sh [comando]"
    echo ""
    echo "Comandos disponíveis:"
    echo "  all         - Executa o pipeline completo"
    echo "  preprocess  - Executa apenas o pipeline de processamento de dados"
    echo "  train       - Executa o pipeline de treinamento de modelos"
    echo "  deploy      - Executa o pipeline de deployment"
    echo "  dashboard   - Inicia o dashboard Streamlit"
    echo "  help        - Exibe esta mensagem de ajuda"
    echo ""
}

# Verificar se o Kedro está instalado
check_kedro() {
    if ! python3 -m kedro --version &> /dev/null; then
        echo "Kedro não encontrado. Instalando dependências..."
        python3 -m pip install -r requirements.txt
    fi
}

# Verificar se a estrutura de diretórios existe
check_directories() {
    # Criar diretórios se não existirem
    mkdir -p data/01_raw
    mkdir -p data/03_primary
    mkdir -p data/05_model_input
    mkdir -p data/06_models
    mkdir -p data/07_model_output
    mkdir -p data/08_reporting
    mkdir -p mlruns
}

# Executar o pipeline completo
run_all() {
    echo "Executando o pipeline completo..."
    python3 -m kedro run
}

# Executar o pipeline de processamento de dados
run_preprocess() {
    echo "Executando o pipeline de processamento de dados..."
    python3 -m kedro run --pipeline=data_processing
}

# Executar o pipeline de treinamento
run_train() {
    echo "Executando o pipeline de treinamento de modelos..."
    python3 -m kedro run --pipeline=model_training
}

# Executar o pipeline de deployment
run_deploy() {
    echo "Executando o pipeline de deployment..."
    python3 -m kedro run --pipeline=model_deployment
}

# Iniciar o dashboard Streamlit
run_dashboard() {
    echo "Iniciando o dashboard Streamlit..."
    python3 -m streamlit run streamlit_app/app.py
}

# Verificar dependências
check_kedro
check_directories

# Processar comandos
case "$1" in
    all)
        run_all
        ;;
    preprocess)
        run_preprocess
        ;;
    train)
        run_train
        ;;
    deploy)
        run_deploy
        ;;
    dashboard)
        run_dashboard
        ;;
    help | *)
        show_help
        ;;
esac