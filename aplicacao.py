"""
Script principal para rodar a aplicação de predição de arremessos de Kobe Bryant.

Este script carrega o modelo treinado e o aplica no conjunto de dados de produção.
"""
import sys
import logging
import pandas as pd
import mlflow
import os
from pathlib import Path
from typing import Dict, Any

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Configurar MLflow
MLFLOW_TRACKING_URI = "mlruns"
os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def carregar_dados(filepath: str) -> pd.DataFrame:
    """
    Carrega os dados de um arquivo parquet.
    
    Args:
        filepath: Caminho para o arquivo
    
    Returns:
        DataFrame carregado
    """
    logging.info(f"Carregando dados de {filepath}")
    return pd.read_parquet(filepath)

def preparar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara os dados para aplicação do modelo.
    
    Args:
        df: DataFrame com dados brutos
    
    Returns:
        DataFrame preparado
    """
    logging.info("Preparando dados para aplicação do modelo...")
    
    # Remover valores faltantes
    df = df.dropna()
    logging.info(f"Dimensões após remover valores faltantes: {df.shape}")
    
    # Selecionar colunas
    colunas_usadas = ["lat", "lng", "minutes_remaining", "period", "playoffs", "shot_distance"]
    if "shot_made_flag" in df.columns:
        colunas_usadas.append("shot_made_flag")
    
    df_filtrado = df[colunas_usadas]
    logging.info(f"Dimensões após selecionar colunas: {df_filtrado.shape}")
    
    return df_filtrado

def carregar_modelo(modelo_path: str) -> Any:
    """
    Carrega o modelo treinado.
    
    Args:
        modelo_path: Caminho para o arquivo do modelo
    
    Returns:
        Modelo carregado
    """
    import pickle
    
    logging.info(f"Carregando modelo de {modelo_path}")
    with open(modelo_path, "rb") as f:
        modelo = pickle.load(f)
    
    return modelo

def aplicar_modelo(modelo: Any, dados: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica o modelo nos dados preparados.
    
    Args:
        modelo: Modelo treinado
        dados: DataFrame com dados preparados
    
    Returns:
        DataFrame com previsões
    """
    logging.info("Aplicando modelo nos dados...")
    
    # Separar features
    colunas_x = ["lat", "lng", "minutes_remaining", "period", "playoffs", "shot_distance"]
    X = dados[colunas_x]
    
    # Fazer previsões
    pred_proba = modelo.predict_proba(X)[:, 1]
    pred_class = modelo.predict(X)
    
    # Adicionar previsões ao DataFrame
    resultado = dados.copy()
    resultado["shot_made_flag_prob"] = pred_proba
    resultado["shot_made_flag_pred"] = pred_class
    
    return resultado

def calcular_metricas(resultados: pd.DataFrame) -> Dict[str, float]:
    """
    Calcula métricas de desempenho se a variável alvo estiver disponível.
    
    Args:
        resultados: DataFrame com previsões
    
    Returns:
        Dicionário com métricas
    """
    from sklearn.metrics import log_loss, f1_score, accuracy_score, confusion_matrix
    
    metricas = {}
    
    if "shot_made_flag" in resultados.columns:
        logging.info("Calculando métricas de desempenho...")
        y_true = resultados["shot_made_flag"]
        y_pred = resultados["shot_made_flag_pred"]
        y_proba = resultados["shot_made_flag_prob"]
        
        metricas["log_loss"] = log_loss(y_true, y_proba)
        metricas["f1_score"] = f1_score(y_true, y_pred)
        metricas["accuracy"] = accuracy_score(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        logging.info(f"Log Loss: {metricas['log_loss']:.4f}")
        logging.info(f"F1 Score: {metricas['f1_score']:.4f}")
        logging.info(f"Acurácia: {metricas['accuracy']:.4f}")
        logging.info(f"Matriz de Confusão:\n{conf_matrix}")
    else:
        logging.info("Variável alvo não disponível. Não é possível calcular métricas.")
    
    return metricas

def salvar_resultados(resultados: pd.DataFrame, output_path: str) -> None:
    """
    Salva os resultados em um arquivo parquet.
    
    Args:
        resultados: DataFrame com resultados
        output_path: Caminho para salvar o arquivo
    """
    logging.info(f"Salvando resultados em {output_path}")
    resultados.to_parquet(output_path)

def run_pipeline() -> None:
    """Executa o pipeline de aplicação."""
    # Iniciar o MLflow run
    with mlflow.start_run(run_name="PipelineAplicacao"):
        try:
            # 1. Carregar dados de produção
            dados_prod = carregar_dados("data/01_raw/dataset_kobe_prod.parquet")
            mlflow.log_metric("registros_producao", len(dados_prod))
            
            # 2. Preparar dados
            dados_prep = preparar_dados(dados_prod)
            mlflow.log_metric("registros_preparados", len(dados_prep))
            
            # 3. Carregar modelo
            modelo = carregar_modelo("data/06_models/modelo_final.pkl")
            
            # 4. Aplicar modelo
            resultados = aplicar_modelo(modelo, dados_prep)
            
            # 5. Calcular métricas (se possível)
            metricas = calcular_metricas(resultados)
            for nome, valor in metricas.items():
                mlflow.log_metric(nome, valor)
            
            # 6. Salvar resultados
            output_path = "data/06_models/predicoes.parquet"
            salvar_resultados(resultados, output_path)
            mlflow.log_artifact(output_path)
            
            logging.info("Pipeline de aplicação concluído com sucesso!")
            
        except Exception as e:
            logging.error(f"Erro na execução do pipeline: {e}")
            raise

if __name__ == "__main__":
    run_pipeline()