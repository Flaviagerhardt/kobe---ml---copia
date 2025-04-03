import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import os
import plotly.express as px
import plotly.graph_objects as go
from mlflow.tracking import MlflowClient

# Configurar página
st.set_page_config(
    page_title="Dashboard de Monitoramento - Kobe Bryant Shots",
    page_icon="🏀",
    layout="wide"
)

# Título da aplicação
st.title("📊 Dashboard de Monitoramento do Modelo")
st.markdown("### Previsão de Arremessos de Kobe Bryant")

# Sidebar para navegação
st.sidebar.title("Navegação")
page = st.sidebar.radio("Ir para", ["Visão Geral", "Desempenho do Modelo", "Monitoramento de Produção", "Distribuição de Dados"])

# Configurar o cliente MLflow
mlflow_tracking_uri = "mlruns"
os.environ["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri
client = MlflowClient(tracking_uri=mlflow_tracking_uri)

# Função para carregar dados
@st.cache_data
def carregar_dados():
    try:
        # Carregar dados de treinamento, teste e produção
        train = pd.read_parquet("data/05_model_input/base_train.parquet")
        test = pd.read_parquet("data/05_model_input/base_test.parquet")
        prod = pd.read_parquet("data/01_raw/dataset_kobe_prod.parquet")
        
        # Carregar previsões
        try:
            predicoes = pd.read_parquet("data/07_model_output/predicoes.parquet")
        except:
            predicoes = None
            
        return train, test, prod, predicoes
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None, None, None, None

# Função para obter métricas do MLflow
@st.cache_data
def obter_metricas_mlflow():
    try:
        # Obter experimentos
        experimentos = client.search_experiments()
        
        # Obter todas as rodadas
        runs = []
        for exp in experimentos:
            exp_runs = client.search_runs(experiment_ids=[exp.experiment_id])
            runs.extend(exp_runs)
        
        # Organizar métricas
        metricas = []
        for run in runs:
            if run.data.metrics:
                metricas.append({
                    "run_id": run.info.run_id,
                    "run_name": run.data.tags.get("mlflow.runName", "Unknown"),
                    "status": run.info.status,
                    "metrics": run.data.metrics,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time
                })
        
        return metricas
    except Exception as e:
        st.error(f"Erro ao obter métricas do MLflow: {e}")
        return []

# Carregar dados
train, test, prod, predicoes = carregar_dados()
metricas_mlflow = obter_metricas_mlflow()

# Página de Visão Geral
if page == "Visão Geral":
    st.header("Visão Geral do Projeto")
    
    # Informações do dataset
    col1, col2, col3 = st.columns(3)
    
    if train is not None and test is not None:
        col1.metric("Tamanho do Conjunto de Treinamento", f"{len(train)}")
        col2.metric("Tamanho do Conjunto de Teste", f"{len(test)}")
        col3.metric("Tamanho do Conjunto de Produção", f"{len(prod) if prod is not None else 'N/A'}")
        
        # Distribuição das classes
        st.subheader("Distribuição da Variável Alvo (shot_made_flag)")
        
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        train_counts = train["shot_made_flag"].value_counts(normalize=True) * 100
        test_counts = test["shot_made_flag"].value_counts(normalize=True) * 100
        
        x = np.arange(2)
        width = 0.35
        
        ax.bar(x - width/2, [train_counts.get(0, 0), train_counts.get(1, 0)], width, label='Treino')
        ax.bar(x + width/2, [test_counts.get(0, 0), test_counts.get(1, 0)], width, label='Teste')
        
        ax.set_xticks(x)
        ax.set_xticklabels(['Errou (0)', 'Acertou (1)'])
        ax.set_ylabel('Porcentagem (%)')
        ax.set_title('Distribuição da Variável Alvo nos Conjuntos de Dados')
        ax.legend()
        
        st.pyplot(fig)
        
        # Mapa de calor de correlação
        st.subheader("Correlação entre Variáveis")
        
        fig = plt.figure(figsize=(10, 8))
        corr = train.corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Mapa de Calor de Correlação")
        
        st.pyplot(fig)
    else:
        st.warning("Não foi possível carregar os dados de treinamento e teste.")

# Página de Desempenho do Modelo
elif page == "Desempenho do Modelo":
    st.header("Desempenho dos Modelos")
    
    # Exibir métricas do MLflow
    if metricas_mlflow:
        # Filtrar apenas runs de treinamento
        runs_treinamento = [m for m in metricas_mlflow if "Treinamento" in m.get("run_name", "")]
        
        if runs_treinamento:
            # Comparação de log loss
            log_loss_data = []
            f1_data = []
            
            for run in runs_treinamento:
                modelo = run["run_name"].replace("Treinamento_", "")
                
                if "log_loss" in run["metrics"]:
                    log_loss_data.append({
                        "Modelo": modelo,
                        "Log Loss": run["metrics"]["log_loss"]
                    })
                
                if "f1_score" in run["metrics"]:
                    f1_data.append({
                        "Modelo": modelo,
                        "F1 Score": run["metrics"]["f1_score"]
                    })
            
            # Gráficos de métricas
            col1, col2 = st.columns(2)
            
            with col1:
                if log_loss_data:
                    st.subheader("Log Loss por Modelo")
                    df_log_loss = pd.DataFrame(log_loss_data)
                    fig = px.bar(df_log_loss, x="Modelo", y="Log Loss", 
                                color="Modelo", title="Log Loss (menor é melhor)")
                    st.plotly_chart(fig)
                
            with col2:
                if f1_data:
                    st.subheader("F1 Score por Modelo")
                    df_f1 = pd.DataFrame(f1_data)
                    fig = px.bar(df_f1, x="Modelo", y="F1 Score", 
                                color="Modelo", title="F1 Score (maior é melhor)")
                    st.plotly_chart(fig)
        else:
            st.warning("Não foram encontradas rodadas de treinamento nos logs do MLflow.")
    else:
        st.warning("Não foi possível obter métricas do MLflow.")

# Página de Monitoramento de Produção
elif page == "Monitoramento de Produção":
    st.header("Monitoramento em Produção")
    
    if predicoes is not None:
        # Distribuição das previsões
        st.subheader("Distribuição das Previsões")
        
        # Histograma de probabilidades preditas
        fig = px.histogram(predicoes, x="shot_made_flag_prob", 
                          title="Distribuição das Probabilidades Preditas",
                          labels={"shot_made_flag_prob": "Probabilidade Predita"})
        st.plotly_chart(fig)
        
        # Se tiver a variável alvo real nos dados de produção
        if "shot_made_flag" in predicoes.columns:
            st.subheader("Comparação: Valores Reais vs. Preditos")
            
            # Matriz de confusão
            conf_matrix = pd.crosstab(predicoes["shot_made_flag"], 
                                     predicoes["shot_made_flag_pred"], 
                                     rownames=['Real'], 
                                     colnames=['Predito'])
            
            fig = px.imshow(conf_matrix, 
                           text_auto=True, 
                           color_continuous_scale='Blues',
                           title="Matriz de Confusão")
            st.plotly_chart(fig)
            
            # Curva ROC
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve(predicoes["shot_made_flag"], predicoes["shot_made_flag_prob"])
            roc_auc = auc(fpr, tpr)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'ROC curve (AUC = {roc_auc:.3f})'))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
            fig.update_layout(
                title='Curva ROC',
                xaxis_title='Taxa de Falsos Positivos',
                yaxis_title='Taxa de Verdadeiros Positivos'
            )
            st.plotly_chart(fig)
            
            # Métricas de desempenho em produção
            col1, col2 = st.columns(2)
            
            # Obter métricas de produção dos logs do MLflow
            prod_metrics = None
            for m in metricas_mlflow:
                if "PipelineAplicacao" in m.get("run_name", ""):
                    prod_metrics = m["metrics"]
                    break
            
            if prod_metrics:
                if "log_loss_producao" in prod_metrics:
                    col1.metric("Log Loss em Produção", f"{prod_metrics['log_loss_producao']:.4f}")
                if "f1_score_producao" in prod_metrics:
                    col2.metric("F1 Score em Produção", f"{prod_metrics['f1_score_producao']:.4f}")
        else:
            st.info("Não há valores reais disponíveis para comparação no conjunto de produção.")
    else:
        st.warning("Os dados de previsão não estão disponíveis.")

# Página de Distribuição de Dados
elif page == "Distribuição de Dados":
    st.header("Distribuição de Dados")
    
    if train is not None and prod is not None:
        # Selecionar variável para visualizar
        features = ["lat", "lng", "minutes_remaining", "period", "playoffs", "shot_distance"]
        feature = st.selectbox("Selecionar variável para visualizar", features)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"Distribuição em Treino - {feature}")
            fig = px.histogram(train, x=feature, color="shot_made_flag", 
                              barmode="overlay", histnorm="percent",
                              labels={"shot_made_flag": "Acertou"})
            st.plotly_chart(fig)
        
        with col2:
            st.subheader(f"Distribuição em Produção - {feature}")
            
            # Se tiver shot_made_flag em prod
            if "shot_made_flag" in prod.columns:
                fig = px.histogram(prod, x=feature, color="shot_made_flag", 
                                  barmode="overlay", histnorm="percent",
                                  labels={"shot_made_flag": "Acertou"})
            else:
                fig = px.histogram(prod, x=feature, histnorm="percent")
            
            st.plotly_chart(fig)
        
        # Verificar drift (mudança na distribuição)
        st.subheader("Análise de Data Drift")
        
        # Calcular estatísticas para comparação
        train_stats = train[feature].describe()
        prod_stats = prod[feature].describe()
        
        # Criar df com estatísticas lado a lado
        stats_df = pd.DataFrame({
            "Estatística": train_stats.index,
            "Treinamento": train_stats.values,
            "Produção": prod_stats.values,
            "Diferença (%)": ((prod_stats.values - train_stats.values) / train_stats.values * 100)
        })
        
        # Destacar diferenças significativas
        def highlight_drift(val):
            if isinstance(val, float) and abs(val) > 10:
                return 'background-color: yellow'
            return ''
        
        st.dataframe(stats_df.style.applymap(highlight_drift, subset=['Diferença (%)']))
        
        # Mostrar análise de drift do MLflow
        st.subheader("Análise de Drift Global")
        
        # Obter logs de análise de drift
        drift_runs = [m for m in metricas_mlflow if "AnaliseDrift" in m.get("run_name", "")]
        
        if drift_runs:
            latest_drift = drift_runs[-1]["metrics"]
            
            # Mostrar features com drift
            features_with_drift = []
            for feature in features:
                if f"has_drift_{feature}" in latest_drift and latest_drift[f"has_drift_{feature}"] == 1:
                    features_with_drift.append({
                        "Feature": feature, 
                        "p-value": latest_drift.get(f"p_value_{feature}", 0),
                        "Diferença Média (%)": latest_drift.get(f"mean_diff_pct_{feature}", 0)
                    })
            
            if features_with_drift:
                st.warning(f"Foram detectados drifts em {len(features_with_drift)} features!")
                st.dataframe(pd.DataFrame(features_with_drift))
            else:
                st.success("Não foram detectados drifts significativos nas features!")
            
            # Mostrar métricas globais
            col1, col2 = st.columns(2)
            col1.metric("Features com Drift", f"{int(latest_drift.get('features_with_drift', 0))}/{len(features)}")
            col2.metric("Percentual de Drift", f"{latest_drift.get('drift_percentage', 0):.1f}%")
            
            if "significant_drift" in latest_drift and latest_drift["significant_drift"] == 1:
                st.error("Foi detectado um drift significativo global! Recomenda-se retreinar o modelo.")
            else:
                st.info("Não foi detectado drift significativo global. O modelo mantém sua validade.")
        else:
            st.info("Não foram encontrados registros de análise de drift no MLflow.")
    else:
        st.warning("Não foi possível carregar os dados de treinamento e produção.")

# Informações adicionais no rodapé
st.markdown("---")
st.markdown("**Dashboard de Monitoramento do Projeto de ML - Arremessos do Kobe Bryant**")
st.markdown("Desenvolvido como parte do Projeto de Disciplina de Engenharia de Machine Learning")