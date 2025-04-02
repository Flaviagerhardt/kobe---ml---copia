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

    train_set = train_set.rename(columns={"lon": "lng"})
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

    train_set = train_set.rename(columns={"lon": "lng"})
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
    # Comparar log loss (menor é melhor)
    logging.info(f"Log Loss - Regressão Logística: {metricas_regressao['log_loss']}")
    logging.info(f"Log Loss - Árvore de Decisão: {metricas_arvore['log_loss']}")
    logging.info(f"F1 Score - Regressão Logística: {metricas_regressao['f1_score']}")
    logging.info(f"F1 Score - Árvore de Decisão: {metricas_arvore['f1_score']}")
    
    # Decidir com base no F1 Score (maior é melhor)
    if metricas_arvore['f1_score'] > metricas_regressao['f1_score']:
        logging.info("Árvore de decisão selecionada como melhor modelo!")
        return modelo_arvore, metricas_arvore
    else:
        logging.info("Regressão logística selecionada como melhor modelo!")
        return modelo_regressao, metricas_regressao

def aplicar_modelo(modelo_final, dados_prod):
    import mlflow
    from sklearn.metrics import log_loss, f1_score

    mlflow.start_run(run_name="PipelineAplicacao", nested=True)

    # Corrigir nome da coluna, se necessário
    if "lon" in dados_prod.columns and "lng" not in dados_prod.columns:
        dados_prod = dados_prod.rename(columns={"lon": "lng"})

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
        else:
            print("Sem registros válidos para avaliação (todos com NaN em shot_made_flag).")
    else:
        print("Coluna 'shot_made_flag' não encontrada. Sem métricas para registrar.")

    mlflow.end_run()

    return resultado