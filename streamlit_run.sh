#!/bin/bash
# Script para iniciar o dashboard Streamlit

# Verificar se o Streamlit está instalado
if ! python3 -m streamlit --version &> /dev/null; then
    echo "Streamlit não encontrado. Instalando..."
    python3 -m pip install streamlit
fi

# Verificar se todas as dependências necessárias estão instaladas
echo "Verificando dependências..."
python3 -m pip install -q pandas numpy matplotlib seaborn mlflow plotly scikit-learn

# Garantir que os diretórios existam
mkdir -p data/01_raw
mkdir -p data/03_primary
mkdir -p data/05_model_input
mkdir -p data/06_models
mkdir -p data/07_model_output
mkdir -p data/08_reporting
mkdir -p mlruns

# Criar diretório para o app Streamlit se não existir
mkdir -p streamlit_app

# Copiar o arquivo do app aprimorado para o diretório correto
echo "Copiando o app Streamlit..."
cp -f app.py streamlit_app/

# Iniciar o Streamlit
echo "Iniciando o dashboard Streamlit..."
cd streamlit_app
python3 -m streamlit run app.py