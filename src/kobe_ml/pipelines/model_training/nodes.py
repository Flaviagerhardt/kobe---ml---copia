import pandas as pd
import numpy as np
import mlflow
import logging
from typing import Dict, Tuple, Any
import os
from sklearn.metrics import log_loss, f1_score
import pickle
import matplotlib.pyplot as plt
import json
from pycaret.classification import setup, create_model, finalize_model, predict_model, plot_model

def treinar_regressao_logistica(train_set: pd.DataFrame, test_set: pd.DataFrame, params: Dict[str, Any]) -> Tuple[object, Dict[str, float]]:
    """
    Treina um modelo de regressão logística com PyCaret e avalia seu desempenho.
    
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
    
    # Inicializar ambiente PyCaret com dados de treinamento
    setup_data = train_set.copy()
    target_col = 'shot_made_flag'
    
    # Temporariamente desativar o MLflow para evitar conflitos
    old_tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri("")
    
    try:
        # Configurar ambiente PyCaret com parâmetros corrigidos (e sem log_experiment)
        clf_setup = setup(
            data=setup_data, 
            target=target_col,
            session_id=params['session_id'],
            normalize=params.get('normalize', True),
            feature_selection=params.get('feature_selection', True),
            pca=params.get('pca', False),
            pca_components=params.get('pca_components', 0.95),
            remove_outliers=params.get('remove_outliers', False),
            polynomial_features=params.get('polynomial_features', False),
            # Remover log_experiment para evitar conflitos com MLflow
            log_experiment=False
        )
        
        # Treinar modelo de regressão logística
        modelo = create_model('lr', fold=5)
        
        # Finalizar o modelo
        modelo_final = finalize_model(modelo)
        
    finally:
        # Restaurar configuração original do MLflow
        mlflow.set_tracking_uri(old_tracking_uri)
    
    # Preparar os dados de teste para avaliação
    X_test = test_set.drop(columns=[target_col])
    y_test = test_set[target_col]
    
    # Fazer previsões no conjunto de teste
    y_pred = predict_model(modelo_final, data=test_set)
    
    # Log das colunas disponíveis para facilitar depuração
    logging.info(f"Colunas disponíveis no resultado do predict_model: {list(y_pred.columns)}")
    
    # Verificar diferentes nomes de colunas que podem estar presentes
    y_pred_class = None
    if 'Label' in y_pred.columns:
        y_pred_class = y_pred['Label'].values
    elif 'prediction_label' in y_pred.columns:
        y_pred_class = y_pred['prediction_label'].values
    
    if y_pred_class is None:
        raise ValueError(f"Coluna de classe predita não encontrada. Colunas disponíveis: {list(y_pred.columns)}")
    
    # Extrair probabilidades - verificar diferentes nomes de colunas
    prob_col = None
    if 'Score_1' in y_pred.columns:
        prob_col = 'Score_1'
    elif 'Score' in y_pred.columns:
        prob_col = 'Score'
    elif 'prediction_score_1' in y_pred.columns:
        prob_col = 'prediction_score_1'
    elif 'prediction_score' in y_pred.columns:
        prob_col = 'prediction_score'
    
    if prob_col:
        y_pred_proba = y_pred[prob_col].values
    else:
        # Se não encontrar coluna de probabilidade, usar previsões de classe (menos preciso)
        logging.warning("Coluna de probabilidade não encontrada. Usando previsões de classe para calcular log_loss.")
        y_pred_proba = y_pred_class
    
    # Calcular métricas
    logloss = log_loss(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred_class)
    
    # Registrar métricas no MLflow
    mlflow.log_metric("log_loss", logloss)
    mlflow.log_metric("f1_score", f1)
    
    # Salvar modelo e registrar como artefato
    os.makedirs("data/06_models", exist_ok=True)
    modelo_path = "data/06_models/modelo_regressao.pkl"
    
    # Salvar modelo com pickle
    with open(modelo_path, 'wb') as f:
        pickle.dump(modelo_final, f)
    
    mlflow.log_artifact(modelo_path)
    
    # Tentar gerar curva ROC
    try:
        # Temporariamente desativar o MLflow para evitar conflitos
        mlflow.set_tracking_uri("")
        
        # Gerar plot
        plot_model(modelo, plot='auc', scale=0.7, save=True)
        
        # Restaurar configuração original do MLflow
        mlflow.set_tracking_uri(old_tracking_uri)
        
        # Encontrar o arquivo salvo pelo PyCaret
        roc_path = "AUC.png"
        if os.path.exists(roc_path):
            # Mover para o diretório correto
            os.makedirs("data/08_reporting", exist_ok=True)
            new_roc_path = "data/08_reporting/roc_lr.png"
            os.rename(roc_path, new_roc_path)
            mlflow.log_artifact(new_roc_path)
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
    
    return modelo_final, metricas

def treinar_arvore_decisao(train_set: pd.DataFrame, test_set: pd.DataFrame, params: Dict[str, Any]) -> Tuple[object, Dict[str, float]]:
    """
    Treina um modelo de árvore de decisão com PyCaret e avalia seu desempenho.
    
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
    
    # Inicializar ambiente PyCaret com dados de treinamento
    setup_data = train_set.copy()
    target_col = 'shot_made_flag'
    
    # Temporariamente desativar o MLflow para evitar conflitos
    old_tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri("")
    
    try:
        # Configurar ambiente PyCaret com parâmetros corrigidos (e sem log_experiment)
        clf_setup = setup(
            data=setup_data, 
            target=target_col,
            session_id=params['session_id'],
            normalize=params.get('normalize', True),
            feature_selection=params.get('feature_selection', True),
            pca=params.get('pca', False),
            pca_components=params.get('pca_components', 0.95),
            remove_outliers=params.get('remove_outliers', False),
            polynomial_features=params.get('polynomial_features', False),
            # Remover log_experiment para evitar conflitos com MLflow
            log_experiment=False
        )
        
        # Treinar modelo de árvore de decisão
        modelo = create_model('dt', fold=5)
        
        # Finalizar o modelo
        modelo_final = finalize_model(modelo)
        
    finally:
        # Restaurar configuração original do MLflow
        mlflow.set_tracking_uri(old_tracking_uri)
    
    # Preparar os dados de teste para avaliação
    X_test = test_set.drop(columns=[target_col])
    y_test = test_set[target_col]
    
    # Fazer previsões no conjunto de teste
    y_pred = predict_model(modelo_final, data=test_set)
    
    # Log das colunas disponíveis para facilitar depuração
    logging.info(f"Colunas disponíveis no resultado do predict_model: {list(y_pred.columns)}")
    
    # Verificar diferentes nomes de colunas que podem estar presentes
    y_pred_class = None
    if 'Label' in y_pred.columns:
        y_pred_class = y_pred['Label'].values
    elif 'prediction_label' in y_pred.columns:
        y_pred_class = y_pred['prediction_label'].values
    
    if y_pred_class is None:
        raise ValueError(f"Coluna de classe predita não encontrada. Colunas disponíveis: {list(y_pred.columns)}")
    
    # Extrair probabilidades - verificar diferentes nomes de colunas
    prob_col = None
    if 'Score_1' in y_pred.columns:
        prob_col = 'Score_1'
    elif 'Score' in y_pred.columns:
        prob_col = 'Score'
    elif 'prediction_score_1' in y_pred.columns:
        prob_col = 'prediction_score_1'
    elif 'prediction_score' in y_pred.columns:
        prob_col = 'prediction_score'
    
    if prob_col:
        y_pred_proba = y_pred[prob_col].values
    else:
        # Se não encontrar coluna de probabilidade, usar previsões de classe (menos preciso)
        logging.warning("Coluna de probabilidade não encontrada. Usando previsões de classe para calcular log_loss.")
        y_pred_proba = y_pred_class
    
    # Calcular métricas
    logloss = log_loss(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred_class)
    
    # Registrar métricas no MLflow
    mlflow.log_metric("log_loss", logloss)
    mlflow.log_metric("f1_score", f1)
    
    # Salvar modelo e registrar como artefato
    os.makedirs("data/06_models", exist_ok=True)
    modelo_path = "data/06_models/modelo_arvore.pkl"
    
    # Salvar modelo com pickle
    with open(modelo_path, 'wb') as f:
        pickle.dump(modelo_final, f)
    
    mlflow.log_artifact(modelo_path)
    
    # Tentar gerar gráficos
    try:
        # Temporariamente desativar o MLflow para evitar conflitos
        mlflow.set_tracking_uri("")
        
        # Gerar plot ROC
        plot_model(modelo, plot='auc', scale=0.7, save=True)
        
        # Encontrar o arquivo salvo pelo PyCaret
        roc_path = "AUC.png"
        if os.path.exists(roc_path):
            # Mover para o diretório correto
            os.makedirs("data/08_reporting", exist_ok=True)
            new_roc_path = "data/08_reporting/roc_dt.png"
            os.rename(roc_path, new_roc_path)
            
            # Restaurar configuração do MLflow para registrar artefato
            mlflow.set_tracking_uri(old_tracking_uri)
            mlflow.log_artifact(new_roc_path)
            
            # Desativar MLflow novamente para próxima operação
            mlflow.set_tracking_uri("")
        
        # Gerar plot de feature importance
        plot_model(modelo, plot='feature', scale=0.7, save=True)
        
        # Restaurar configuração original do MLflow
        mlflow.set_tracking_uri(old_tracking_uri)
        
        # Encontrar o arquivo salvo pelo PyCaret
        feat_path = "Feature Importance.png"
        if os.path.exists(feat_path):
            # Mover para o diretório correto
            os.makedirs("data/08_reporting", exist_ok=True)
            feat_imp_path = "data/08_reporting/feature_importance.png"
            os.rename(feat_path, feat_imp_path)
            mlflow.log_artifact(feat_imp_path)
    except Exception as e:
        logging.warning(f"Não foi possível gerar gráficos: {e}")
        # Garantir que o MLflow está restaurado
        mlflow.set_tracking_uri(old_tracking_uri)
    
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
    
    return modelo_final, metricas

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
        mlflow.log_metric("modelo_selecionado", 1)  # 1 para árvore de decisão
        
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
        mlflow.log_metric("modelo_selecionado", 0)  # 0 para regressão logística
        
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