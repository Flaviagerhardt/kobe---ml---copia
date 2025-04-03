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
import pickle
from sklearn.metrics import log_loss, f1_score, accuracy_score, confusion_matrix

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
    
    # Renomear coluna "lon" para "lng" se necessário
    if "lon" in df.columns and "lng" not in df.columns:
        df = df.rename(columns={"lon": "lng"})
    
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
    
    resultado = dados.copy()
    
    # Verificar se estamos usando um modelo PyCaret
    try:
        # Tentar primeiro como modelo PyCaret
        from pycaret.classification import predict_model
        
        # Aplicar modelo utilizando o PyCaret
        try:
            # Tente primeiro com raw_score=True 
            try:
                predictions = predict_model(modelo, data=dados, raw_score=True)
                logging.info("PyCaret predict_model chamado com raw_score=True")
            except:
                # Se falhar, tente sem o parâmetro raw_score
                predictions = predict_model(modelo, data=dados)
                logging.info("PyCaret predict_model chamado sem raw_score")
            
            # Extrair previsões - verificar diferentes formatos de coluna para compatibilidade
            pred_col = None
            if 'Label' in predictions.columns:
                pred_col = 'Label'
                resultado['shot_made_flag_pred'] = predictions['Label']
            elif 'prediction_label' in predictions.columns:
                pred_col = 'prediction_label'
                resultado['shot_made_flag_pred'] = predictions['prediction_label']
            
            if not pred_col:
                raise ValueError("Coluna de previsão não encontrada nos resultados do PyCaret")
                
            # Extrair probabilidades - verificar diferentes formatos de coluna
            prob_col = None
            if 'Score_1' in predictions.columns:
                prob_col = 'Score_1'
                resultado['shot_made_flag_prob'] = predictions['Score_1']
            elif 'Score' in predictions.columns:
                prob_col = 'Score'
                resultado['shot_made_flag_prob'] = predictions['Score']
            elif 'prediction_score_1' in predictions.columns:
                prob_col = 'prediction_score_1'
                resultado['shot_made_flag_prob'] = predictions['prediction_score_1']
            elif 'prediction_score' in predictions.columns:
                prob_col = 'prediction_score'
                resultado['shot_made_flag_prob'] = predictions['prediction_score']
            
            if not prob_col:
                logging.warning("Coluna de probabilidade não encontrada nos resultados do PyCaret. Usando valores binários.")
                resultado['shot_made_flag_prob'] = resultado['shot_made_flag_pred']
            
            logging.info(f"Modelo aplicado usando PyCaret. Colunas encontradas: {list(predictions.columns)}")
            return resultado
            
        except Exception as e:
            logging.warning(f"Erro ao usar PyCaret para previsão: {e}. Tentando método alternativo...")
    except ImportError:
        logging.warning("PyCaret não encontrado. Tentando método alternativo...")
    
    # Método alternativo: usar scikit-learn diretamente
    try:
        # Separar features
        colunas_x = ["lat", "lng", "minutes_remaining", "period", "playoffs", "shot_distance"]
        X = dados[colunas_x]
        
        # Usando scikit-learn predict diretamente
        resultado["shot_made_flag_prob"] = modelo.predict_proba(X)[:, 1]
        resultado["shot_made_flag_pred"] = modelo.predict(X)
        
        logging.info("Modelo aplicado usando scikit-learn")
        
    except Exception as e:
        logging.error(f"Erro ao aplicar modelo: {e}")
        raise
    
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
    # Garantir que o diretório existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
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
            output_path = "data/07_model_output/predicoes.parquet"
            salvar_resultados(resultados, output_path)
            mlflow.log_artifact(output_path)
            
            logging.info("Pipeline de aplicação concluído com sucesso!")
            
        except Exception as e:
            logging.error(f"Erro na execução do pipeline: {e}")
            raise

if __name__ == "__main__":
    run_pipeline()