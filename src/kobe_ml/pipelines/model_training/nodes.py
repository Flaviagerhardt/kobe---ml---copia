import pandas as pd
import numpy as np
import mlflow
import logging
from typing import Dict, Tuple, Any
import os
from pycaret.classification import *
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, f1_score

def treinar_regressao_logistica(train_set: pd.DataFrame, test_set: pd.DataFrame, params: Dict[str, Any]) -> Tuple[object, Dict[str, float]]:
    """
    Treina um modelo de regressão logística usando PyCaret e avalia seu desempenho.
    
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
    
    # Filtrar parâmetros válidos para o setup do PyCaret
    setup_params = {
        'data': train_set,
        'target': 'shot_made_flag',
        'session_id': params['session_id'],
        'normalize': params['normalize']
    }
    
    # Adicionar parâmetros opcionais disponíveis na versão do PyCaret
    if 'feature_selection' in params:
        setup_params['feature_selection'] = params['feature_selection']
    
    if 'pca' in params:
        setup_params['pca'] = params['pca']
    
    if 'pca_components' in params and params['pca']:
        setup_params['pca_components'] = params['pca_components']
    
    if 'ignore_low_variance' in params:
        setup_params['ignore_low_variance'] = params['ignore_low_variance']
    
    # Configurar ambiente PyCaret
    clf = setup(**setup_params, log_experiment=True, silent=True, verbose=False)
    
    # Criar e treinar o modelo de regressão logística
    modelo = create_model('lr', verbose=False)
    
    # Avaliar o modelo
    try:
        eval_resultados = evaluate_model(modelo, return_dict=True)
    except:
        logging.warning("Não foi possível obter resultados detalhados da avaliação do modelo.")
    
    # Fazer previsões no conjunto de teste
    predictions = predict_model(modelo, data=test_set)
    
    # Extrair as previsões de probabilidade e classe
    y_pred = predictions['prediction_label'] if 'prediction_label' in predictions.columns else predictions['Label']
    
    # Para probabilidades, ajustar com base nas colunas disponíveis
    if 'prediction_score' in predictions.columns:
        y_pred_proba = predictions['prediction_score']
    elif 'Score_1' in predictions.columns:
        y_pred_proba = predictions['Score_1']
    else:
        # Se não conseguir obter probabilidades diretamente, fazer novo predict com score
        try:
            predictions_proba = predict_model(modelo, data=test_set, raw_score=True)
            y_pred_proba = predictions_proba['Score_1']
        except:
            logging.warning("Não foi possível obter probabilidades do modelo.")
            # Usar valores binários como aproximação
            y_pred_proba = y_pred
    
    # Calcular métricas adicionais
    logloss = log_loss(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    
    # Registrar métricas no MLflow
    mlflow.log_metric("log_loss", logloss)
    mlflow.log_metric("f1_score", f1)
    
    # Salvar modelo e registrar como artefato
    modelo_finalizado = finalize_model(modelo)
    
    # Garantir que o diretório existe
    os.makedirs("data/06_models", exist_ok=True)
    
    modelo_path = "data/06_models/modelo_regressao.pkl"
    with open(modelo_path, 'wb') as f:
        pickle.dump(modelo_finalizado, f)
    
    mlflow.log_artifact(modelo_path)
    
    # Tentar salvar gráficos importantes, com tratamento de erro
    try:
        os.makedirs("data/08_reporting", exist_ok=True)
        
        # Curva ROC
        roc_plot = plot_model(modelo, plot='auc', save=True)
        if os.path.exists('AUC.png'):
            mlflow.log_artifact('AUC.png')
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
    
    import json
    with open("data/08_reporting/metricas_regressao.json", 'w') as f:
        json.dump(metricas, f)
    
    return modelo_finalizado, metricas

def treinar_arvore_decisao(train_set: pd.DataFrame, test_set: pd.DataFrame, params: Dict[str, Any]) -> Tuple[object, Dict[str, float]]:
    """
    Treina um modelo de árvore de decisão usando PyCaret e avalia seu desempenho.
    
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
    
    # Verificar se já existe uma configuração PyCaret
    pycaret_setup_exists = False
    try:
        # Verificar se o ambiente do PyCaret já está configurado
        current_exp = get_current_experiment()
        if current_exp is not None:
            pycaret_setup_exists = True
            logging.info("Usando configuração PyCaret existente")
    except:
        pycaret_setup_exists = False
    
    # Configurar o ambiente PyCaret se necessário
    if not pycaret_setup_exists:
        # Filtrar parâmetros válidos para o setup do PyCaret
        setup_params = {
            'data': train_set,
            'target': 'shot_made_flag',
            'session_id': params['session_id'],
            'normalize': params['normalize']
        }
        
        # Adicionar parâmetros opcionais disponíveis na versão do PyCaret
        if 'feature_selection' in params:
            setup_params['feature_selection'] = params['feature_selection']
        
        if 'pca' in params:
            setup_params['pca'] = params['pca']
        
        if 'pca_components' in params and params['pca']:
            setup_params['pca_components'] = params['pca_components']
        
        if 'ignore_low_variance' in params:
            setup_params['ignore_low_variance'] = params['ignore_low_variance']
        
        # Configurar ambiente PyCaret
        clf = setup(**setup_params, log_experiment=True, silent=True, verbose=False)
    
    # Criar e treinar o modelo de árvore de decisão
    modelo = create_model('dt', verbose=False)
    
    # Avaliar o modelo
    try:
        eval_resultados = evaluate_model(modelo, return_dict=True)
    except:
        logging.warning("Não foi possível obter resultados detalhados da avaliação do modelo.")
    
    # Fazer previsões no conjunto de teste
    predictions = predict_model(modelo, data=test_set)
    
    # Extrair as previsões de probabilidade e classe
    y_pred = predictions['prediction_label'] if 'prediction_label' in predictions.columns else predictions['Label']
    
    # Para probabilidades, ajustar com base nas colunas disponíveis
    if 'prediction_score' in predictions.columns:
        y_pred_proba = predictions['prediction_score']
    elif 'Score_1' in predictions.columns:
        y_pred_proba = predictions['Score_1']
    else:
        # Se não conseguir obter probabilidades diretamente, fazer novo predict com score
        try:
            predictions_proba = predict_model(modelo, data=test_set, raw_score=True)
            y_pred_proba = predictions_proba['Score_1']
        except:
            logging.warning("Não foi possível obter probabilidades do modelo.")
            # Usar valores binários como aproximação
            y_pred_proba = y_pred
    
    # Calcular métricas adicionais
    logloss = log_loss(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    
    # Registrar métricas no MLflow
    mlflow.log_metric("log_loss", logloss)
    mlflow.log_metric("f1_score", f1)
    
    # Salvar modelo e registrar como artefato
    modelo_finalizado = finalize_model(modelo)
    
    # Garantir que o diretório existe
    os.makedirs("data/06_models", exist_ok=True)
    
    modelo_path = "data/06_models/modelo_arvore.pkl"
    with open(modelo_path, 'wb') as f:
        pickle.dump(modelo_finalizado, f)
    
    mlflow.log_artifact(modelo_path)
    
    # Tentar salvar gráficos importantes, com tratamento de erro
    try:
        os.makedirs("data/08_reporting", exist_ok=True)
        
        # Curva ROC
        roc_plot = plot_model(modelo, plot='auc', save=True)
        if os.path.exists('AUC.png'):
            mlflow.log_artifact('AUC.png')
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
    
    import json
    with open("data/08_reporting/metricas_arvore.json", 'w') as f:
        json.dump(metricas, f)
    
    return modelo_finalizado, metricas

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