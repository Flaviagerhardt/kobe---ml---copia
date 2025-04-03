import pandas as pd
import numpy as np
import mlflow
import logging
from typing import Dict, Tuple, Any
import os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss, f1_score
import pickle
import matplotlib.pyplot as plt
import json  # Adicionando a importação do módulo json

def treinar_regressao_logistica(train_set: pd.DataFrame, test_set: pd.DataFrame, params: Dict[str, Any]) -> Tuple[object, Dict[str, float]]:
    """
    Treina um modelo de regressão logística com scikit-learn e avalia seu desempenho.
    
    Args:
        train_set: DataFrame com dados de treinamento
        test_set: DataFrame com dados de teste
        params: Dicionário com parâmetros do modelo
    
    Returns:
        Modelo treinado e dicionário com métricas
    """
    # Configurar MLflow
    mlflow.start_run(run_name="Treinamento_RegressaoLogistica", nested=True)
    
    # Registrar parâmetros do modelo
    for key, value in params.items():
        mlflow.log_param(key, value)
    
    # Preparar os dados
    X_train = train_set.drop(columns=["shot_made_flag"])
    y_train = train_set["shot_made_flag"]
    X_test = test_set.drop(columns=["shot_made_flag"])
    y_test = test_set["shot_made_flag"]
    
    # Treinar modelo de regressão logística
    modelo = LogisticRegression(random_state=params['session_id'], max_iter=1000)
    modelo.fit(X_train, y_train)
    
    # Fazer previsões no conjunto de teste
    y_pred = modelo.predict(X_test)
    y_pred_proba = modelo.predict_proba(X_test)[:, 1]
    
    # Calcular métricas
    logloss = log_loss(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    
    # Registrar métricas no MLflow
    mlflow.log_metric("log_loss", logloss)
    mlflow.log_metric("f1_score", f1)
    
    # Salvar modelo e registrar como artefato
    os.makedirs("data/06_models", exist_ok=True)
    modelo_path = "data/06_models/modelo_regressao.pkl"
    with open(modelo_path, 'wb') as f:
        pickle.dump(modelo, f)
    
    mlflow.log_artifact(modelo_path)
    
    # Tentar gerar curva ROC
    try:
        from sklearn.metrics import RocCurveDisplay
        import matplotlib.pyplot as plt
        os.makedirs("data/08_reporting", exist_ok=True)
        
        # Curva ROC
        roc_plot = RocCurveDisplay.from_estimator(modelo, X_test, y_test)
        plt.title('Curva ROC - Regressão Logística')
        roc_path = "data/08_reporting/roc_lr.png"
        plt.savefig(roc_path)
        plt.close()
        mlflow.log_artifact(roc_path)
    except Exception as e:
        logging.warning(f"Não foi possível gerar gráficos: {e}")
    
    # Finalizar o MLflow run
    mlflow.end_run()
    
    # Preparar dicionário de métricas
    metricas = {
        "modelo": "regressao_logistica",
        "log_loss": logloss,
        "f1_score": f1
    }
    
    # Salvar métricas como JSON
    os.makedirs("data/08_reporting", exist_ok=True)
    
    with open("data/08_reporting/metricas_regressao.json", 'w') as f:
        json.dump(metricas, f)
    
    return modelo, metricas

def treinar_arvore_decisao(train_set: pd.DataFrame, test_set: pd.DataFrame, params: Dict[str, Any]) -> Tuple[object, Dict[str, float]]:
    """
    Treina um modelo de árvore de decisão com scikit-learn e avalia seu desempenho.
    
    Args:
        train_set: DataFrame com dados de treinamento
        test_set: DataFrame com dados de teste
        params: Dicionário com parâmetros do modelo
    
    Returns:
        Modelo treinado e dicionário com métricas
    """
    # Configurar MLflow
    mlflow.start_run(run_name="Treinamento_ArvoreDecisao", nested=True)
    
    # Registrar parâmetros do modelo
    for key, value in params.items():
        mlflow.log_param(key, value)
    
    # Preparar os dados
    X_train = train_set.drop(columns=["shot_made_flag"])
    y_train = train_set["shot_made_flag"]
    X_test = test_set.drop(columns=["shot_made_flag"])
    y_test = test_set["shot_made_flag"]
    
    # Treinar modelo de árvore de decisão
    modelo = DecisionTreeClassifier(random_state=params['session_id'])
    modelo.fit(X_train, y_train)
    
    # Fazer previsões no conjunto de teste
    y_pred = modelo.predict(X_test)
    y_pred_proba = modelo.predict_proba(X_test)[:, 1]
    
    # Calcular métricas
    logloss = log_loss(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    
    # Registrar métricas no MLflow
    mlflow.log_metric("log_loss", logloss)
    mlflow.log_metric("f1_score", f1)
    
    # Salvar modelo e registrar como artefato
    os.makedirs("data/06_models", exist_ok=True)
    modelo_path = "data/06_models/modelo_arvore.pkl"
    with open(modelo_path, 'wb') as f:
        pickle.dump(modelo, f)
    
    mlflow.log_artifact(modelo_path)
    
    # Tentar gerar gráficos
    try:
        from sklearn.metrics import RocCurveDisplay
        import matplotlib.pyplot as plt
        os.makedirs("data/08_reporting", exist_ok=True)
        
        # Curva ROC
        roc_plot = RocCurveDisplay.from_estimator(modelo, X_test, y_test)
        plt.title('Curva ROC - Árvore de Decisão')
        roc_path = "data/08_reporting/roc_dt.png"
        plt.savefig(roc_path)
        plt.close()
        mlflow.log_artifact(roc_path)
        
        # Importância das features
        importances = modelo.feature_importances_
        features = X_train.columns
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 6))
        plt.title('Importância das Features - Árvore de Decisão')
        plt.bar(range(X_train.shape[1]), importances[indices], align='center')
        plt.xticks(range(X_train.shape[1]), features[indices], rotation=90)
        plt.tight_layout()
        feat_imp_path = "data/08_reporting/feature_importance.png"
        plt.savefig(feat_imp_path)
        plt.close()
        mlflow.log_artifact(feat_imp_path)
    except Exception as e:
        logging.warning(f"Não foi possível gerar gráficos: {e}")
    
    # Finalizar o MLflow run
    mlflow.end_run()
    
    # Preparar dicionário de métricas
    metricas = {
        "modelo": "arvore_decisao",
        "log_loss": logloss,
        "f1_score": f1
    }
    
    # Salvar métricas como JSON
    os.makedirs("data/08_reporting", exist_ok=True)
    
    with open("data/08_reporting/metricas_arvore.json", 'w') as f:
        json.dump(metricas, f)
    
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
    
    # Comparar métricas
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
        
        # Copiar modelo para arquivo final
        modelo_final_path = "data/06_models/modelo_final.pkl"
        with open("data/06_models/modelo_arvore.pkl", 'rb') as f_src:
            with open(modelo_final_path, 'wb') as f_dst:
                f_dst.write(f_src.read())
        
        mlflow.log_artifact(modelo_final_path)
        
        # Salvar métricas do modelo final
        with open("data/08_reporting/metricas_final.json", 'w') as f:
            json.dump(metricas_arvore, f)
        
        # Finalizar o MLflow run
        mlflow.end_run()
        
        return modelo_arvore, metricas_arvore
    else:
        logging.info("Regressão logística selecionada como melhor modelo!")
        mlflow.log_param("modelo_selecionado", "regressao_logistica")
        
        # Copiar modelo para arquivo final
        modelo_final_path = "data/06_models/modelo_final.pkl"
        with open("data/06_models/modelo_regressao.pkl", 'rb') as f_src:
            with open(modelo_final_path, 'wb') as f_dst:
                f_dst.write(f_src.read())
        
        mlflow.log_artifact(modelo_final_path)
        
        # Salvar métricas do modelo final
        with open("data/08_reporting/metricas_final.json", 'w') as f:
            json.dump(metricas_regressao, f)
        
        # Finalizar o MLflow run
        mlflow.end_run()
        
        return modelo_regressao, metricas_regressao