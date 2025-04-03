import pandas as pd
import numpy as np
import mlflow
import logging
from typing import Dict, Tuple, Any
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss, f1_score
import pickle

def treinar_regressao_logistica(train_set: pd.DataFrame, test_set: pd.DataFrame) -> Tuple[object, Dict[str, float]]:
    """
    Treina um modelo de regressão logística e avalia seu desempenho.
    
    Args:
        train_set: DataFrame com dados de treinamento
        test_set: DataFrame com dados de teste
    
    Returns:
        Modelo treinado e dicionário com métricas
    """
    # Configurar MLflow
    mlflow.start_run(run_name="Treinamento_RegressaoLogistica", nested=True)

    # Verificar se é necessário renomear colunas
    if "lon" in train_set.columns and "lng" not in train_set.columns:
        train_set = train_set.rename(columns={"lon": "lng"})
    if "lon" in test_set.columns and "lng" not in test_set.columns:
        test_set = test_set.rename(columns={"lon": "lng"})
    
    # Separar features e target
    X_train = train_set.drop(columns=["shot_made_flag"])
    y_train = train_set["shot_made_flag"]
    X_test = test_set.drop(columns=["shot_made_flag"])
    y_test = test_set["shot_made_flag"]
    
    # Treinar modelo
    modelo = LogisticRegression(random_state=42)
    modelo.fit(X_train, y_train)
    
    # Fazer previsões
    y_pred_proba = modelo.predict_proba(X_test)[:, 1]
    y_pred = modelo.predict(X_test)
    
    # Calcular métricas
    logloss = log_loss(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    
    # Registrar métricas no MLflow
    mlflow.log_metric("log_loss", logloss)
    mlflow.log_metric("f1_score", f1)
    
    # Registrar o modelo
    mlflow.sklearn.log_model(modelo, "modelo_regressao_logistica")
    
    # Finalizar o MLflow run
    mlflow.end_run()
    
    # Preparar dicionário de métricas
    metricas = {
        "modelo": "regressao_logistica",
        "log_loss": logloss,
        "f1_score": f1
    }
    
    return modelo, metricas

def treinar_arvore_decisao(train_set: pd.DataFrame, test_set: pd.DataFrame) -> Tuple[object, Dict[str, float]]:
    """
    Treina um modelo de árvore de decisão e avalia seu desempenho.
    
    Args:
        train_set: DataFrame com dados de treinamento
        test_set: DataFrame com dados de teste
    
    Returns:
        Modelo treinado e dicionário com métricas
    """
    # Configurar MLflow
    mlflow.start_run(run_name="Treinamento_ArvoreDecisao", nested=True)

    # Verificar se é necessário renomear colunas
    if "lon" in train_set.columns and "lng" not in train_set.columns:
        train_set = train_set.rename(columns={"lon": "lng"})
    if "lon" in test_set.columns and "lng" not in test_set.columns:
        test_set = test_set.rename(columns={"lon": "lng"})
    
    # Separar features e target
    X_train = train_set.drop(columns=["shot_made_flag"])
    y_train = train_set["shot_made_flag"]
    X_test = test_set.drop(columns=["shot_made_flag"])
    y_test = test_set["shot_made_flag"]
    
    # Treinar modelo
    modelo = DecisionTreeClassifier(random_state=42)
    modelo.fit(X_train, y_train)
    
    # Fazer previsões
    y_pred_proba = modelo.predict_proba(X_test)[:, 1]
    y_pred = modelo.predict(X_test)
    
    # Calcular métricas
    logloss = log_loss(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    
    # Registrar métricas no MLflow
    mlflow.log_metric("log_loss", logloss)
    mlflow.log_metric("f1_score", f1)
    
    # Registrar o modelo
    mlflow.sklearn.log_model(modelo, "modelo_arvore_decisao")
    
    # Finalizar o MLflow run
    mlflow.end_run()
    
    # Preparar dicionário de métricas
    metricas = {
        "modelo": "arvore_decisao",
        "log_loss": logloss,
        "f1_score": f1
    }
    
    return modelo, metricas

def selecionar_melhor_modelo(modelo_regressao: object, metricas_regressao: Dict[str, float],
                           modelo_arvore: object, metricas_arvore: Dict[str, float]) -> Tuple[object, Dict[str, float]]:
    """
    Seleciona o melhor modelo com base nas métricas de avaliação.
    
    Args:
        modelo_regressao: Modelo de regressão logística treinado
        metricas_regressao: Métricas do modelo de regressão
        modelo_arvore: Modelo de árvore de decisão treinado
        metricas_arvore: Métricas do modelo de árvore
    
    Returns:
        Melhor modelo e suas métricas
    """
    # Configurar MLflow
    mlflow.start_run(run_name="SelecaoModelo", nested=True)
    
    # Comparar log loss (menor é melhor)
    logging.info(f"Log Loss - Regressão Logística: {metricas_regressao['log_loss']}")
    logging.info(f"Log Loss - Árvore de Decisão: {metricas_arvore['log_loss']}")
    logging.info(f"F1 Score - Regressão Logística: {metricas_regressao['f1_score']}")
    logging.info(f"F1 Score - Árvore de Decisão: {metricas_arvore['f1_score']}")
    
    # Registrar métricas dos dois modelos para comparação
    mlflow.log_metric("log_loss_regressao", metricas_regressao['log_loss'])
    mlflow.log_metric("f1_score_regressao", metricas_regressao['f1_score'])
    mlflow.log_metric("log_loss_arvore", metricas_arvore['log_loss'])
    mlflow.log_metric("f1_score_arvore", metricas_arvore['f1_score'])
    
    # Decidir com base no F1 Score (maior é melhor)
    if metricas_arvore['f1_score'] > metricas_regressao['f1_score']:
        logging.info("Árvore de decisão selecionada como melhor modelo!")
        mlflow.log_param("modelo_selecionado", "arvore_decisao")
        
        # Registrar o modelo final
        mlflow.sklearn.log_model(modelo_arvore, "modelo_final")
        
        # Finalizar o MLflow run
        mlflow.end_run()
        
        return modelo_arvore, metricas_arvore
    else:
        logging.info("Regressão logística selecionada como melhor modelo!")
        mlflow.log_param("modelo_selecionado", "regressao_logistica")
        
        # Registrar o modelo final
        mlflow.sklearn.log_model(modelo_regressao, "modelo_final")
        
        # Finalizar o MLflow run
        mlflow.end_run()
        
        return modelo_regressao, metricas_regressao

def aplicar_modelo(modelo_final, dados_prod):
    """
    Aplica o modelo selecionado aos dados de produção.
    
    Args:
        modelo_final: Modelo treinado e selecionado
        dados_prod: DataFrame com dados de produção
    
    Returns:
        DataFrame com previsões
    """
    import mlflow
    from sklearn.metrics import log_loss, f1_score

    mlflow.start_run(run_name="PipelineAplicacao", nested=True)

    # Corrigir nome da coluna, se necessário
    if "lon" in dados_prod.columns and "lng" not in dados_prod.columns:
        dados_prod = dados_prod.rename(columns={"lon": "lng"})

    # Remover linhas com dados faltantes
    dados_prod = dados_prod.dropna()

    # Colunas usadas no modelo
    colunas_usadas = ["lat", "lng", "minutes_remaining", "period", "playoffs", "shot_distance"]

    # Validação das colunas
    for col in colunas_usadas:
        if col not in dados_prod.columns:
            raise ValueError(f"A coluna '{col}' não está presente nos dados de produção.")

    # Previsões
    X_prod = dados_prod[colunas_usadas]
    y_pred_proba = modelo_final.predict_proba(X_prod)[:, 1]
    y_pred = modelo_final.predict(X_prod)

    # Criar DataFrame de resultados
    resultado = dados_prod.copy()
    resultado["probabilidade_acerto"] = y_pred_proba
    resultado["previsao"] = y_pred

    # Verificar se existe a variável alvo para avaliação
    if "shot_made_flag" in dados_prod.columns:
        resultado["shot_made_flag"] = dados_prod["shot_made_flag"]

        # Eliminar registros sem rótulo para calcular métricas
        resultado_avaliacao = resultado.dropna(subset=["shot_made_flag"])

        if not resultado_avaliacao.empty:
            y_true = resultado_avaliacao["shot_made_flag"]
            y_pred_proba_filtrado = resultado_avaliacao["probabilidade_acerto"]
            y_pred_filtrado = resultado_avaliacao["previsao"]

            logloss = log_loss(y_true, y_pred_proba_filtrado)
            f1 = f1_score(y_true, y_pred_filtrado)

            # Logar métricas no MLflow
            mlflow.log_metric("log_loss_producao", logloss)
            mlflow.log_metric("f1_score_producao", f1)
            
            logging.info(f"Log Loss em produção: {logloss:.4f}")
            logging.info(f"F1 Score em produção: {f1:.4f}")
        else:
            logging.info("Sem registros válidos para avaliação (todos com NaN em shot_made_flag).")
    else:
        logging.info("Coluna 'shot_made_flag' não encontrada. Sem métricas para registrar.")

    # Salvar previsões como artefato
    temp_file = "data/07_model_output/predicoes_temp.parquet"
    resultado.to_parquet(temp_file)
    mlflow.log_artifact(temp_file)
    
    mlflow.end_run()

    return resultado