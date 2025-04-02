import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
import logging

def preparar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara os dados removendo valores faltantes e selecionando colunas específicas.
    
    Args:
        df: DataFrame com os dados brutos
    
    Returns:
        DataFrame filtrado
    """
    # Iniciar MLflow run
    mlflow.start_run(run_name="PreparacaoDados", nested=True)
    
    # Dimensões originais
    dimensao_original = df.shape
    mlflow.log_metric("linhas_original", dimensao_original[0])
    mlflow.log_metric("colunas_original", dimensao_original[1])
    
    # Remover dados faltantes
    df = df.dropna()
    logging.info(f"Dimensões após remover valores faltantes: {df.shape}")
    
    # Correção: renomear 'lon' para 'lng' para manter consistência com o código
    df = df.rename(columns={"lon": "lng"})
    
    # Selecionar colunas
    colunas_usadas = ["lat", "lng", "minutes_remaining", "period", "playoffs", "shot_distance", "shot_made_flag"]
    df_filtrado = df[colunas_usadas]
    
    # Dimensões após filtragem
    dimensao_final = df_filtrado.shape
    mlflow.log_metric("linhas_filtrado", dimensao_final[0])
    mlflow.log_metric("colunas_filtrado", dimensao_final[1])
    
    # Salvar dados filtrados em arquivo temporário para MLflow no caminho correto
    temp_file = "data/03_primary/data_filtered.parquet"

    # Garantir que o diretório exista
    import os
    os.makedirs("data/03_primary", exist_ok=True)

    df_filtrado.to_parquet(temp_file)

    # Registrar arquivo como artefato
    mlflow.log_artifact(temp_file)
    
    # Finalizar MLflow run
    mlflow.end_run()
    
    logging.info(f"Dimensões finais do dataset filtrado: {dimensao_final}")
    return df_filtrado

def splitar_dados(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Separa os dados em conjuntos de treinamento e teste.
    
    Args:
        df: DataFrame com os dados preparados
        test_size: Proporção do conjunto de teste
        random_state: Semente aleatória para reprodutibilidade
    
    Returns:
        DataFrames de treinamento e teste
    """
    # Iniciar MLflow run
    mlflow.start_run(run_name="SplitDados", nested=True)
    
    # Registrar parâmetros
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("random_state", random_state)
    
    X = df.drop(columns=["shot_made_flag"])
    y = df["shot_made_flag"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=random_state
    )

    train = X_train.copy()
    train["shot_made_flag"] = y_train

    test = X_test.copy()
    test["shot_made_flag"] = y_test
    
    # Registrar métricas
    mlflow.log_metric("tamanho_treino", len(train))
    mlflow.log_metric("tamanho_teste", len(test))
    
    # Calcular distribuição de classes
    mlflow.log_metric("positivos_treino", train["shot_made_flag"].sum())
    mlflow.log_metric("positivos_teste", test["shot_made_flag"].sum())
    
    # Salvar datasets em arquivos temporários nas pastas corretas
    temp_train = "data/05_model_input/base_train.parquet"
    temp_test = "data/05_model_input/base_test.parquet"

    # Garantir que os diretórios existam
    import os
    os.makedirs("data/05_model_input", exist_ok=True)

    train.to_parquet(temp_train)
    test.to_parquet(temp_test)
    
    # Registrar artefatos
    mlflow.log_artifact(temp_train)
    mlflow.log_artifact(temp_test)
    
    # Finalizar MLflow run
    mlflow.end_run()
    
    logging.info(f"Tamanho do conjunto de treinamento: {len(train)}")
    logging.info(f"Tamanho do conjunto de teste: {len(test)}")
    
    return train, test

def inspecionar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inspeciona os dados e retorna o mesmo DataFrame.
    
    Args:
        df: DataFrame com os dados
        
    Returns:
        DataFrame original
    """
    # Apenas loga algumas informações sobre o DataFrame
    logging.info(f"Dimensões do DataFrame: {df.shape}")
    logging.info(f"Colunas do DataFrame: {df.columns.tolist()}")
    
    # Verificar dados faltantes
    missing = df.isnull().sum()
    if missing.sum() > 0:
        logging.info(f"Valores faltantes por coluna:\n{missing[missing > 0]}")
    else:
        logging.info("Não há valores faltantes no DataFrame.")
        
    return df