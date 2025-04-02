"""
Nós para o pipeline de deployment e monitoramento do modelo.
"""
import pandas as pd
import numpy as np
import mlflow
import logging
from typing import Dict, Any, Optional
from sklearn.metrics import log_loss, f1_score, accuracy_score, confusion_matrix
import scipy.stats as stats

def carregar_modelo(modelo: Any) -> Any:
    """
    Carrega o modelo treinado.
    
    Args:
        modelo: Modelo já carregado pelo Kedro.
        
    Returns:
        O modelo carregado.
    """
    logging.info("Modelo carregado com sucesso")
    return modelo

def preparar_dados_producao(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara os dados de produção para aplicação do modelo.
    
    Args:
        df: DataFrame com os dados de produção.
        
    Returns:
        DataFrame com os dados preparados.
    """
    # Configurar MLflow
    mlflow.start_run(run_name="PreparacaoDadosProducao", nested=True)

    df = df.rename(columns={"lon": "lng"})
    
    # Remover valores faltantes
    df_clean = df.dropna()
    
    # Registrar métricas
    mlflow.log_metric("total_linhas_prod_original", len(df))
    mlflow.log_metric("total_linhas_prod_limpo", len(df_clean))
    mlflow.log_metric("percentual_dados_removidos", 100 * (1 - len(df_clean) / len(df)))
    
    # Selecionar colunas
    colunas_usadas = ["lat", "lng", "minutes_remaining", "period", "playoffs", "shot_distance"]
    if "shot_made_flag" in df_clean.columns:
        colunas_usadas.append("shot_made_flag")
        
    df_preparado = df_clean[colunas_usadas]
    
    # Finalizar MLflow run
    mlflow.end_run()
    
    logging.info(f"Dados de produção preparados: {df_preparado.shape}")
    return df_preparado

def aplicar_modelo(modelo: Any, dados: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica o modelo nos dados de produção.
    
    Args:
        modelo: Modelo treinado.
        dados: DataFrame com dados preparados.
        
    Returns:
        DataFrame com as previsões.
    """
    # Configurar MLflow
    mlflow.start_run(run_name="AplicacaoModeloProducao", nested=True)
    
    # Separar features
    colunas_x = ["lat", "lng", "minutes_remaining", "period", "playoffs", "shot_distance"]
    X = dados[colunas_x]
    
    # Fazer previsões
    y_pred_proba = modelo.predict_proba(X)[:, 1]
    y_pred = modelo.predict(X)
    
    # Criar dataframe com resultados
    resultados = dados.copy()
    resultados["shot_made_flag_prob"] = y_pred_proba
    resultados["shot_made_flag_pred"] = y_pred
    
    # Registrar distribuição das previsões
    mlflow.log_metric("media_probabilidades", y_pred_proba.mean())
    mlflow.log_metric("mediana_probabilidades", np.median(y_pred_proba))
    mlflow.log_metric("percentual_positivos", y_pred.mean() * 100)
    
    # Finalizar MLflow run
    mlflow.end_run()
    
    logging.info(f"Modelo aplicado nos dados de produção. Previsões positivas: {y_pred.sum()} ({y_pred.mean() * 100:.2f}%)")
    return resultados

def salvar_predicoes(resultados: pd.DataFrame) -> pd.DataFrame:
    """
    Salva as predições em um arquivo.
    
    Args:
        resultados: DataFrame com os resultados da predição.
        
    Returns:
        O mesmo DataFrame com os resultados.
    """
    logging.info("Salvando predições do modelo")
    return resultados

def calcular_metricas_producao(resultados: pd.DataFrame, dados_prod: pd.DataFrame) -> Dict[str, float]:
    """
    Calcula métricas de desempenho em produção se a variável alvo estiver disponível.
    
    Args:
        resultados: DataFrame com os resultados da predição.
        dados_prod: DataFrame com os dados de produção.
        
    Returns:
        Dicionário com as métricas calculadas.
    """
    # Configurar MLflow
    mlflow.start_run(run_name="MetricasProducao", nested=True)
    
    metricas = {}
    
    # Verificar se a variável alvo está disponível
    if "shot_made_flag" in dados_prod.columns:
        logging.info("Calculando métricas de desempenho em produção")
        
        y_true = dados_prod["shot_made_flag"]
        y_pred = resultados["shot_made_flag_pred"]
        y_proba = resultados["shot_made_flag_prob"]
        
        # Calcular métricas
        metricas["log_loss_producao"] = log_loss(y_true, y_proba)
        metricas["f1_score_producao"] = f1_score(y_true, y_pred)
        metricas["accuracy_producao"] = accuracy_score(y_true, y_pred)
        
        # Registrar métricas no MLflow
        for nome, valor in metricas.items():
            mlflow.log_metric(nome, valor)
            
        # Calcular matriz de confusão
        conf_matrix = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = conf_matrix.ravel()
        
        metricas["true_negatives"] = int(tn)
        metricas["false_positives"] = int(fp)
        metricas["false_negatives"] = int(fn)
        metricas["true_positives"] = int(tp)
        
        # Registrar métricas adicionais
        metricas["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metricas["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Log de métricas adicionais
        mlflow.log_metric("precision_producao", metricas["precision"])
        mlflow.log_metric("recall_producao", metricas["recall"])
        
        logging.info(f"Log Loss em produção: {metricas['log_loss_producao']:.4f}")
        logging.info(f"F1 Score em produção: {metricas['f1_score_producao']:.4f}")
        logging.info(f"Acurácia em produção: {metricas['accuracy_producao']:.4f}")
    else:
        logging.info("Variável alvo não disponível nos dados de produção. Não é possível calcular métricas de desempenho.")
    
    # Finalizar MLflow run
    mlflow.end_run()
    
    return metricas

def analisar_data_drift(train_set: pd.DataFrame, dados_prod: pd.DataFrame) -> Dict[str, Any]:
    """
    Analisa o drift entre os dados de treinamento e produção.
    
    Args:
        train_set: DataFrame com os dados de treinamento.
        dados_prod: DataFrame com os dados de produção.
        
    Returns:
        Dicionário com os resultados da análise de drift.
    """
    # Configurar MLflow
    mlflow.start_run(run_name="AnaliseDrift", nested=True)
    
    # Inicializar dicionário para resultados
    drift_results = {}
    
    # Selecionar colunas para análise
    features = ["lat", "lng", "minutes_remaining", "period", "playoffs", "shot_distance"]
    
    # Para cada feature, calcular p-value do teste KS
    for feature in features:
        train_values = train_set[feature].values
        prod_values = dados_prod[feature].values
        
        # Calcular estatísticas descritivas
        train_mean = train_values.mean()
        prod_mean = prod_values.mean()
        train_std = train_values.std()
        prod_std = prod_values.std()
        
        # Registrar métricas de estatísticas descritivas
        mlflow.log_metric(f"train_mean_{feature}", train_mean)
        mlflow.log_metric(f"prod_mean_{feature}", prod_mean)
        mlflow.log_metric(f"train_std_{feature}", train_std)
        mlflow.log_metric(f"prod_std_{feature}", prod_std)
        
        # Calcular a diferença relativa nas médias
        mean_diff_pct = abs((prod_mean - train_mean) / train_mean * 100) if train_mean != 0 else np.inf
        mlflow.log_metric(f"mean_diff_pct_{feature}", mean_diff_pct)
        
        # Teste KS para detectar drift
        ks_statistic, p_value = stats.ks_2samp(train_values, prod_values)
        
        # Registrar resultados do teste KS
        mlflow.log_metric(f"ks_statistic_{feature}", ks_statistic)
        mlflow.log_metric(f"p_value_{feature}", p_value)
        
        # Determinar se há drift (p-value < 0.05 sugere distribuições diferentes)
        
        has_drift = bool(p_value < 0.05)  # <-- força tipo Python nativo
        mlflow.log_metric(f"has_drift_{feature}", int(has_drift))
        
        # Armazenar resultados
        drift_results[feature] = {
            "train_mean": train_mean,
            "prod_mean": prod_mean,
            "train_std": train_std,
            "prod_std": prod_std,
            "mean_diff_pct": mean_diff_pct,
            "ks_statistic": ks_statistic,
            "p_value": p_value,
            "has_drift": has_drift
        }
        
        logging.info(f"Feature {feature}: {'Drift detectado!' if has_drift else 'Sem drift significativo.'} (p-value: {p_value:.4f})")
    
    # Verificar drift global (quantas features apresentam drift)
    features_with_drift = sum(1 for f in features if drift_results[f]["has_drift"])
    drift_results["features_with_drift"] = features_with_drift
    drift_results["total_features"] = len(features)
    drift_results["drift_percentage"] = features_with_drift / len(features) * 100
    
    # Registrar métricas globais
    mlflow.log_metric("features_with_drift", features_with_drift)
    mlflow.log_metric("drift_percentage", drift_results["drift_percentage"])
    
    # Determinar se há drift significativo global
    significant_drift = drift_results["drift_percentage"] > 30  # Se mais de 30% das features têm drift
    drift_results["significant_drift"] = bool(significant_drift)
    mlflow.log_metric("significant_drift", int(significant_drift))
    
    # Finalizar MLflow run
    mlflow.end_run()
    
    logging.info(f"Análise de drift concluída. {features_with_drift}/{len(features)} features com drift ({drift_results['drift_percentage']:.1f}%).")
    logging.info(f"Drift significativo global: {'SIM' if significant_drift else 'NÃO'}")
    
    return drift_results