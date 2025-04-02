#!/bin/bash
# Script para iniciar o dashboard Streamlit

# Verificar se o Streamlit está instalado
if ! python3 -m streamlit --version &> /dev/null; then
    echo "Streamlit não encontrado. Instalando..."
    python3 -m pip install streamlit
fi

# Criar diretório para o app Streamlit se não existir
mkdir -p streamlit_app

# Copiar o arquivo do app para o diretório correto
cp -f src/app.py streamlit_app/

# Iniciar o Streamlit
echo "Iniciando o dashboard Streamlit..."
cd streamlit_app
python3 -m streamlit run app.py