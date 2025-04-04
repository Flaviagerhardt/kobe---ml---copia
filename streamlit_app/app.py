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
import sys
from pathlib import Path
import pickle
from scipy import stats

# Configurar página
st.set_page_config(
    page_title="Dashboard de Monitoramento - Kobe Bryant Shots",
    page_icon="🏀",
    layout="wide"
)

# Ajustar caminhos - verificar se estamos executando do diretório raiz ou de streamlit_app
CURRENT_DIR = Path(os.getcwd())
ROOT_DIR = CURRENT_DIR
if CURRENT_DIR.name == "streamlit_app":
    ROOT_DIR = CURRENT_DIR.parent

# Configurar caminhos relativos
DATA_DIR = ROOT_DIR / "data"
MLFLOW_DIR = ROOT_DIR / "mlruns"

# Título da aplicação
st.title("📊 Dashboard de Monitoramento do Modelo")
st.markdown("### Previsão de Arremessos de Kobe Bryant")

# Sidebar para navegação
st.sidebar.title("Navegação")
page = st.sidebar.radio("Ir para", ["Visão Geral", "Desempenho do Modelo", "Monitoramento de Produção", "Análise de Drift", "Retreinamento"])

# Exibir informações de debug se necessário
show_debug = st.sidebar.checkbox("Mostrar Informações de Debug", False)
if show_debug:
    st.sidebar.write(f"Diretório Atual: {CURRENT_DIR}")
    st.sidebar.write(f"Diretório Raiz: {ROOT_DIR}")
    st.sidebar.write(f"Diretório de Dados: {DATA_DIR}")
    st.sidebar.write(f"Diretório MLflow: {MLFLOW_DIR}")
    st.sidebar.write(f"Arquivos em diretório data (se existir):")
    if DATA_DIR.exists():
        for path in DATA_DIR.glob("**/*.parquet"):
            st.sidebar.write(f"- {path.relative_to(DATA_DIR)}")

# Configurar o cliente MLflow
mlflow_tracking_uri = str(MLFLOW_DIR)
os.environ["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri

try:
    client = MlflowClient(tracking_uri=mlflow_tracking_uri)
except Exception as e:
    st.warning(f"Erro ao configurar MLflow: {e}")
    client = None

# Função para carregar dados
@st.cache_data
def carregar_dados():
    resultados = {}
    erros = {}
    
    # Lista de arquivos para tentar carregar
    arquivos = {
        "train": DATA_DIR / "05_model_input" / "base_train.parquet",
        "test": DATA_DIR / "05_model_input" / "base_test.parquet",
        "prod": DATA_DIR / "01_raw" / "dataset_kobe_prod.parquet",
        "predicoes": DATA_DIR / "07_model_output" / "predicoes.parquet"
    }
    
    # Tentar carregar cada arquivo
    for nome, caminho in arquivos.items():
        try:
            if caminho.exists():
                resultados[nome] = pd.read_parquet(caminho)
                
                # Verificar se precisamos renomear a coluna 'lon' para 'lng'
                if 'lon' in resultados[nome].columns and 'lng' not in resultados[nome].columns:
                    resultados[nome] = resultados[nome].rename(columns={'lon': 'lng'})
            else:
                erros[nome] = f"Arquivo não encontrado: {caminho}"
        except Exception as e:
            erros[nome] = str(e)
    
    return resultados, erros

# Função para obter métricas do MLflow
@st.cache_data
def obter_metricas_mlflow():
    if client is None:
        return []
        
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

# Função para carregar modelos
@st.cache_resource
def carregar_modelos():
    modelos = {}
    erros = {}
    
    # Lista de modelos para tentar carregar
    arquivos_modelo = {
        "modelo_final": DATA_DIR / "06_models" / "modelo_final.pkl",
        "modelo_regressao": DATA_DIR / "06_models" / "modelo_regressao.pkl",
        "modelo_arvore": DATA_DIR / "06_models" / "modelo_arvore.pkl"
    }
    
    # Tentar carregar cada modelo
    for nome, caminho in arquivos_modelo.items():
        try:
            if caminho.exists():
                # Usar pickle para carregar o modelo
                with open(caminho, 'rb') as f:
                    modelos[nome] = pickle.load(f)
            else:
                erros[nome] = f"Arquivo não encontrado: {caminho}"
        except Exception as e:
            erros[nome] = str(e)
    
    return modelos, erros

# Função para analisar drift
def analisar_drift(train_df, prod_df, feature):
    # Verificar se a feature existe em ambos os DataFrames
    if feature not in train_df.columns:
        st.error(f"A feature '{feature}' não existe no DataFrame de treino.")
        return None
    if feature not in prod_df.columns:
        st.error(f"A feature '{feature}' não existe no DataFrame de produção.")
        return None
    
    train_values = train_df[feature].values
    prod_values = prod_df[feature].values
    
    # Calcular estatísticas
    train_mean = train_values.mean()
    prod_mean = prod_values.mean() 
    train_std = train_values.std()
    prod_std = prod_values.std()
    
    # Calcular diferença relativa nas médias
    mean_diff_pct = abs((prod_mean - train_mean) / train_mean * 100) if train_mean != 0 else float('inf')
    
    # Teste KS para detectar drift
    ks_statistic, p_value = stats.ks_2samp(train_values, prod_values)
    
    # Determinar se há drift (p-value < 0.05 sugere distribuições diferentes)
    has_drift = bool(p_value < 0.05)
    
    return {
        "train_mean": train_mean,
        "prod_mean": prod_mean,
        "train_std": train_std,
        "prod_std": prod_std,
        "mean_diff_pct": mean_diff_pct,
        "ks_statistic": ks_statistic,
        "p_value": p_value,
        "has_drift": has_drift
    }

# Verificar se os diretórios de dados existem
if not DATA_DIR.exists():
    st.error(f"Diretório de dados não encontrado: {DATA_DIR}")
    st.info("Execute os pipelines Kedro para gerar os dados antes de usar o dashboard.")
    st.stop()

# Carregar dados e modelos
dados, erros_carregamento = carregar_dados()
modelos, erros_modelos = carregar_modelos()
metricas_mlflow = obter_metricas_mlflow()

# Exibir erros de carregamento se houver
if erros_carregamento and show_debug:
    st.sidebar.subheader("Erros de Carregamento:")
    for nome, erro in erros_carregamento.items():
        st.sidebar.error(f"Erro ao carregar {nome}: {erro}")

if erros_modelos and show_debug:
    st.sidebar.subheader("Erros ao carregar modelos:")
    for nome, erro in erros_modelos.items():
        st.sidebar.error(f"Erro ao carregar {nome}: {erro}")

# Verificar se temos os dados mínimos necessários
dados_minimos = "train" in dados and "test" in dados
if not dados_minimos:
    st.warning("Dados de treinamento e/ou teste não encontrados.")
    st.info("Execute os pipelines de processamento de dados e treinamento antes de usar o dashboard.")
    st.info("Use o comando: `./run.sh all` ou `./run.sh preprocess` seguido de `./run.sh train`")
    st.stop()

# Extrair dados para facilitar o acesso
train = dados.get("train")
test = dados.get("test")
prod = dados.get("prod")
predicoes = dados.get("predicoes")

# Página de Visão Geral
if page == "Visão Geral":
    st.header("Visão Geral do Projeto")
    
    st.markdown("""
    ### Sobre o projeto
    Este dashboard monitora um modelo de machine learning que prevê se os arremessos de Kobe Bryant resultaram em cestas ou não.
    O pipeline foi construído usando Kedro para orquestração  e MLflow para rastreamento de experimentos.
    """)
    
    # Informações do dataset
    col1, col2, col3 = st.columns(3)
    
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
    
    # Distribuição das features
    st.subheader("Distribuição das Features Principais")
    
    feature = st.selectbox("Selecionar feature para visualizar", 
                         ["shot_distance", "period", "minutes_remaining", "playoffs"])
    
    fig = px.histogram(train, x=feature, color="shot_made_flag", 
                      barmode="group",
                      labels={"shot_made_flag": "Acertou (1) / Errou (0)"},
                      title=f"Distribuição de {feature} por resultado do arremesso")
    st.plotly_chart(fig)

# Página de Desempenho do Modelo
elif page == "Desempenho do Modelo":
    st.header("Desempenho dos Modelos")
    
    st.markdown("""
    ### Comparação de Modelos
    Nesta seção, comparamos o desempenho dos dois modelos treinados:
    - Regressão Logística
    - Árvore de Decisão
    """)
    
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
            
            # Mostrar informações do modelo selecionado
            st.subheader("Informações do Modelo Final")
            
            # Verificar qual modelo foi selecionado
            modelo_selecionado = None
            for run in metricas_mlflow:
                if "SelecaoModelo" in run.get("run_name", ""):
                    if "modelo_selecionado" in run["metrics"]:
                        modelo_selecionado = run["metrics"]["modelo_selecionado"]
                    break
            
            if modelo_selecionado:
                st.info(f"**Modelo selecionado**: {modelo_selecionado}")
                st.write("O modelo foi selecionado com base no F1 Score, que equilibra precisão e recall.")
            
            # Mostrar exemplos de previsões
            if test is not None and "modelo_final" in modelos:
                st.subheader("Exemplos de Previsões no Conjunto de Teste")
                
                try:
                    # Usar scikit-learn para fazer previsões em algumas amostras
                    # Usar apenas primeiras 10 amostras para demonstração
                    test_amostra = test.head(10).copy()
                    X_test_amostra = test_amostra.drop(columns=["shot_made_flag"])
                    y_test_amostra = test_amostra["shot_made_flag"]
                    
                    # Fazer previsões
                    modelo = modelos["modelo_final"]
                    y_pred = modelo.predict(X_test_amostra)
                    y_proba = modelo.predict_proba(X_test_amostra)[:, 1]
                    
                    # Criar dataframe para visualização
                    preds_view = test_amostra.copy()
                    preds_view["Probabilidade"] = y_proba
                    preds_view["Previsão"] = y_pred
                    preds_view["Real"] = y_test_amostra
                    preds_view["Correto"] = preds_view["Previsão"] == preds_view["Real"]
                    
                    # Mostrar a tabela de previsões
                    st.dataframe(preds_view[["shot_distance", "period", "playoffs", "Real", "Previsão", "Probabilidade", "Correto"]])
                except Exception as e:
                    st.error(f"Erro ao fazer previsões: {e}")
        else:
            st.warning("Não foram encontradas rodadas de treinamento nos logs do MLflow.")
    else:
        st.warning("Logs do MLflow não encontrados. Execute o pipeline de treinamento para gerar métricas.")

# Página de Monitoramento de Produção
elif page == "Monitoramento de Produção":
    st.header("Monitoramento em Produção")
    
    if predicoes is not None:
        # Distribuição das previsões
        st.subheader("Distribuição das Previsões")
        
        # Histograma de probabilidades preditas
        prob_col = None
        if "shot_made_flag_prob" in predicoes.columns:
            prob_col = "shot_made_flag_prob"
        elif "probabilidade_acerto" in predicoes.columns:
            prob_col = "probabilidade_acerto"
        elif "Score_1" in predicoes.columns:
            prob_col = "Score_1"
            
        if prob_col:
            fig = px.histogram(predicoes, x=prob_col, 
                            title="Distribuição das Probabilidades Preditas",
                            labels={prob_col: "Probabilidade Predita"},
                            nbins=20)
            st.plotly_chart(fig)
        else:
            st.warning("Coluna de probabilidades não encontrada nos dados de previsão.")
        
        # Se tiver a variável alvo real nos dados de produção
        pred_col = None
        if "shot_made_flag_pred" in predicoes.columns:
            pred_col = "shot_made_flag_pred"
        elif "previsao" in predicoes.columns:
            pred_col = "previsao"
        elif "Label" in predicoes.columns:
            pred_col = "Label"
        
        if "shot_made_flag" in predicoes.columns and pred_col:
            st.subheader("Comparação: Valores Reais vs. Preditos")
            
            # Matriz de confusão
            conf_matrix = pd.crosstab(predicoes["shot_made_flag"], 
                                    predicoes[pred_col], 
                                    rownames=['Real'], 
                                    colnames=['Predito'])
            
            fig = px.imshow(conf_matrix, 
                        text_auto=True, 
                        color_continuous_scale='Blues',
                        title="Matriz de Confusão")
            st.plotly_chart(fig)
            
            # Curva ROC
            from sklearn.metrics import roc_curve, auc
            if prob_col:
                fpr, tpr, _ = roc_curve(predicoes["shot_made_flag"], predicoes[prob_col])
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
            col1, col2, col3 = st.columns(3)
            
            # Calcular métricas atuais
            from sklearn.metrics import log_loss, f1_score, accuracy_score
            try:
                ll = log_loss(predicoes["shot_made_flag"], predicoes[prob_col])
                f1 = f1_score(predicoes["shot_made_flag"], predicoes[pred_col])
                acc = accuracy_score(predicoes["shot_made_flag"], predicoes[pred_col])
                
                col1.metric("Log Loss em Produção", f"{ll:.4f}")
                col2.metric("F1 Score em Produção", f"{f1:.4f}")
                col3.metric("Acurácia em Produção", f"{acc:.4f}")
            except Exception as e:
                st.error(f"Erro ao calcular métricas: {e}")
            
            # Mostrar previsões mais confiantes/erradas
            st.subheader("Análise de Erros")
            
            try:
                # Criar coluna com o erro (0 se acertou, 1 se errou)
                predicoes_erro = predicoes.copy()
                predicoes_erro["erro"] = (predicoes_erro["shot_made_flag"] != predicoes_erro[pred_col]).astype(int)
                
                # Previsões erradas com maior confiança
                st.write("Previsões erradas com maior confiança:")
                erros_confiantes = predicoes_erro[predicoes_erro["erro"] == 1].sort_values(by=prob_col, ascending=False).head(5)
                
                # Selecionar colunas para exibição
                colunas_exibir = ["shot_distance", "period", "playoffs"]
                if "shot_made_flag" in erros_confiantes.columns:
                    colunas_exibir.append("shot_made_flag")
                if pred_col in erros_confiantes.columns:
                    colunas_exibir.append(pred_col)
                if prob_col in erros_confiantes.columns:
                    colunas_exibir.append(prob_col)
                
                st.dataframe(erros_confiantes[colunas_exibir])
            except Exception as e:
                st.error(f"Erro ao analisar erros: {e}")
        else:
            st.info("Não há valores reais disponíveis para comparação no conjunto de produção.")
    else:
        st.warning("Os dados de previsão não estão disponíveis. Execute o pipeline de deployment para gerar previsões.")

# Página de Análise de Drift
elif page == "Análise de Drift":
    st.header("Análise de Data Drift")
    
    if train is not None and prod is not None:
        st.markdown("""
        ### O que é Data Drift?
        O data drift ocorre quando a distribuição estatística dos dados em produção diverge 
        da distribuição dos dados de treinamento. Isso pode degradar o desempenho do modelo 
        ao longo do tempo.
        """)
        
        # Verificar colunas disponíveis em ambos os dataframes
        colunas_train = set(train.columns)
        colunas_prod = set(prod.columns)
        colunas_comuns = colunas_train.intersection(colunas_prod)
        
        # Remover shot_made_flag da lista, se presente
        features = [col for col in colunas_comuns if col != 'shot_made_flag']
        
        # Selecionar variável para visualizar
        feature = st.selectbox("Selecionar variável para analisar", features)
        
        # Analisar drift para a feature selecionada
        drift_result = analisar_drift(train, prod, feature)
        
        if drift_result:
            # Mostrar resultados da análise de drift
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"Distribuição em Treino vs. Produção - {feature}")
                
                df_combined = pd.DataFrame({
                    "Treino": train[feature].sample(min(1000, len(train))),
                    "Produção": prod[feature].sample(min(1000, len(prod)))
                })
                
                # Gráfico de densidade para comparar distribuições
                fig = plt.figure(figsize=(10, 6))
                sns.kdeplot(data=df_combined, fill=True, alpha=0.5)
                plt.title(f"Comparação de Distribuições - {feature}")
                plt.grid(True, alpha=0.3)
                st.pyplot(fig)
                
            with col2:
                st.subheader("Métricas de Drift")
                
                # Formatar diferença de médias
                mean_diff_formatted = f"{drift_result['mean_diff_pct']:.2f}%"
                
                # Mostrar métricas com cores baseadas na significância
                drift_color = "red" if drift_result["has_drift"] else "green"
                
                st.markdown(f"""
                * **Média em Treino:** {drift_result['train_mean']:.4f}
                * **Média em Produção:** {drift_result['prod_mean']:.4f}
                * **Diferença Relativa:** {mean_diff_formatted}
                * **p-value (Teste KS):** <span style='color:{drift_color};'>{drift_result['p_value']:.6f}</span>
                * **Drift Detectado:** <span style='color:{drift_color};'>{drift_result['has_drift']}</span>
                """, unsafe_allow_html=True)
                
                # Boxplots para comparar distribuições
                fig = plt.figure(figsize=(8, 6))
                data = [train[feature], prod[feature]]
                plt.boxplot(data, labels=['Treino', 'Produção'])
                plt.title(f"Boxplot de {feature}")
                plt.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            # Análise de drift global
            st.subheader("Análise de Drift Global")
            
            # Calcular drift para todas as features
            drift_global = {}
            features_com_drift = 0
            
            for f in features:
                # Verificar se a feature existe em ambos os DataFrames
                if f in train.columns and f in prod.columns:
                    result = analisar_drift(train, prod, f)
                    if result:
                        drift_global[f] = result
                        if result["has_drift"]:
                            features_com_drift += 1
            
            # Mostrar resumo
            if drift_global:
                drift_percentage = (features_com_drift / len(drift_global)) * 100
                significant_drift = drift_percentage > 30  # Se mais de 30% das features têm drift
                
                col1, col2 = st.columns(2)
                col1.metric("Features com Drift", f"{features_com_drift}/{len(drift_global)}")
                col2.metric("Percentual de Drift", f"{drift_percentage:.1f}%")
                
                # Recomendação baseada no drift global
                if significant_drift:
                    st.error("Foi detectado um drift significativo global! Recomenda-se retreinar o modelo.")
                else:
                    st.success("Não foi detectado drift significativo global. O modelo mantém sua validade.")
                
                # Tabela com resultados de drift para todas as features
                drift_table = []
                for f, result in drift_global.items():
                    drift_table.append({
                        "Feature": f,
                        "p-value": result["p_value"],
                        "Diferença Média (%)": result["mean_diff_pct"],
                        "Drift Detectado": result["has_drift"]
                    })
                
                st.dataframe(pd.DataFrame(drift_table))
            else:
                st.warning("Não foi possível calcular drift para as features disponíveis.")
        else:
            st.warning(f"Não foi possível analisar drift para a feature {feature}.")
    else:
        st.warning("Não foi possível carregar os dados de treinamento e produção.")

# Página de Retreinamento
elif page == "Retreinamento":
    st.header("Estratégias de Retreinamento")
    
    st.markdown("""
    ### Monitoramento da Saúde do Modelo
    
    #### Cenário com disponibilidade da variável resposta
    Quando temos feedback sobre os resultados reais (se o arremesso foi convertido ou não):
    
    - **Métricas diretas**: Podemos calcular log_loss, F1-score, precisão e recall
    - **Matriz de confusão**: Visualizamos falsos positivos e falsos negativos
    - **Curva ROC e AUC**: Avaliamos a capacidade discriminativa do modelo
    - **Análise temporal**: Monitoramos o desempenho ao longo do tempo para identificar degradação
    
    #### Cenário sem disponibilidade da variável resposta
    Quando não temos feedback imediato sobre os resultados:
    
    - **Monitoramento de distribuições**: Comparamos a distribuição das previsões atuais com históricas
 #### Cenário sem disponibilidade da variável resposta
    Quando não temos feedback imediato sobre os resultados:
    
    - **Monitoramento de distribuições**: Comparamos a distribuição das previsões atuais com históricas
    - **Detecção de drift nas features**: Usamos testes estatísticos para identificar mudanças nas distribuições
    - **Estabilidade das previsões**: Monitoramos variações bruscas na proporção de classes preditas
    - **Feedback indireto**: Relacionamos métricas de negócio (ex: pontos marcados pelo time) com as previsões
    """)
    
    # Separar em abas
    tab1, tab2 = st.tabs(["Estratégia Reativa", "Estratégia Preditiva"])
    
    with tab1:
        st.subheader("Estratégia Reativa de Retreinamento")
        
        st.markdown("""
        Na estratégia reativa, o retreinamento do modelo é acionado quando detectamos sinais de degradação:
        
        1. **Monitoramento contínuo**: Acompanhamos métricas como log_loss e F1-score
        2. **Definição de limiares**: Estabelecemos limites mínimos aceitáveis de desempenho
        3. **Detecção de drift**: Quando identificamos mudanças significativas nos dados de entrada
        4. **Alerta e acionamento**: Sistema notifica a equipe sobre necessidade de retreinamento
        
        ##### Vantagens:
        - Economia de recursos (retreinamos apenas quando necessário)
        - Evita retreinamentos desnecessários quando o modelo mantém bom desempenho
        
        ##### Desvantagens:
        - Pode haver período com desempenho sub-ótimo até a detecção
        - Requer monitoramento constante e confiável
        """)
        
        # Simulação de limiar para retreinamento
        st.subheader("Simulação de Limiar para Retreinamento")
        
        limiar_f1 = st.slider("Limiar mínimo de F1-Score", min_value=0.5, max_value=0.95, value=0.6, step=0.01)
        
        # Obter métricas atuais se disponíveis
        f1_atual = None
        
        if predicoes is not None and "shot_made_flag" in predicoes.columns:
            # Identificar colunas de previsão
            pred_col = None
            if "shot_made_flag_pred" in predicoes.columns:
                pred_col = "shot_made_flag_pred"
            elif "previsao" in predicoes.columns:
                pred_col = "previsao"

            # Calcular F1 score atual
            if pred_col:
                from sklearn.metrics import f1_score
                f1_atual = f1_score(predicoes["shot_made_flag"], predicoes[pred_col])
                
                # Mostrar status baseado no limiar
                col1, col2 = st.columns(2)
                col1.metric("F1-Score Atual", f"{f1_atual:.4f}")
                
                if f1_atual < limiar_f1:
                    col2.error("Retreinamento Necessário!")
                    st.warning(f"O F1-Score atual ({f1_atual:.4f}) está abaixo do limiar ({limiar_f1:.4f}). Recomenda-se retreinar o modelo.")
                else:
                    col2.success("Modelo Saudável")
                    st.info(f"O F1-Score atual ({f1_atual:.4f}) está acima do limiar ({limiar_f1:.4f}). O modelo mantém bom desempenho.")
        else:
            st.info("Sem dados de produção com variável alvo disponível para simular o retreinamento.")
    
    with tab2:
        st.subheader("Estratégia Preditiva de Retreinamento")
        
        st.markdown("""
        Na estratégia preditiva, adotamos uma abordagem proativa, realizando retreinamentos periódicos:
        
        1. **Programação regular**: Retreinamento em intervalos predefinidos (semanal, mensal)
        2. **Shadow models**: Mantemos um modelo "sombra" treinado com dados mais recentes para comparação
        3. **Aprendizado online**: Adaptamos gradualmente o modelo com novos dados
        4. **AutoML periódico**: Otimizamos automaticamente hiperparâmetros durante o retreinamento
        
        ##### Vantagens:
        - Evita degradação prolongada do desempenho
        - Incorpora tendências recentes nos dados
        - Processo mais estruturado e previsível
        
        ##### Desvantagens:
        - Maior consumo de recursos (tempo, computação)
        - Pode reintroduzir novos problemas a cada retreinamento
        """)
        
        # Simulação de calendário de retreinamento
        st.subheader("Simulação de Calendário de Retreinamento")
        
        from datetime import datetime, timedelta
        
        # Data atual
        data_atual = datetime.now()
        
        # Opções de frequência
        frequencia = st.radio("Frequência de Retreinamento", ["Semanal", "Quinzenal", "Mensal", "Trimestral"])
        
        # Calcular próximas datas de retreinamento
        proximas_datas = []
        
        if frequencia == "Semanal":
            dias_intervalo = 7
        elif frequencia == "Quinzenal":
            dias_intervalo = 15
        elif frequencia == "Mensal":
            dias_intervalo = 30
        else:  # Trimestral
            dias_intervalo = 90
        
        # Calcular próximas 5 datas
        for i in range(1, 6):
            proxima_data = data_atual + timedelta(days=i * dias_intervalo)
            proximas_datas.append(proxima_data.strftime("%d/%m/%Y"))
        
        # Mostrar calendário
        st.write("Próximas datas programadas para retreinamento:")
        for i, data in enumerate(proximas_datas):
            st.write(f"{i+1}. {data}")
        
        # Simulação de evolução de métricas
        st.subheader("Simulação de Evolução de Métricas")
        
        # Gerar dados simulados para entender como seria a evolução do F1-score ao longo do tempo
        # com diferentes estratégias de retreinamento
        
        import numpy as np
        
        # Datas para simulação (últimos 12 meses)
        datas = [(data_atual - timedelta(days=30*i)).strftime("%b/%Y") for i in range(12, 0, -1)]
        
        # Simular F1-score com degradação gradual para estratégia reativa
        np.random.seed(42)
        f1_base = 0.70
        f1_reativo = [max(0.5, f1_base - i*0.01 + np.random.normal(0, 0.01)) for i in range(12)]
        
        # Simular F1-score para estratégia preditiva (retreinamento a cada 3 meses)
        f1_preditivo = []
        for i in range(12):
            if i % 3 == 0:  # Retreinamento trimestral
                f1_preditivo.append(f1_base + np.random.normal(0, 0.01))
            else:
                f1_preditivo.append(max(0.5, f1_preditivo[-1] - 0.01 + np.random.normal(0, 0.01)))
        
        # Criar DataFrame para visualização
        df_simulacao = pd.DataFrame({
            "Data": datas,
            "F1-Score (Reativo)": f1_reativo,
            "F1-Score (Preditivo)": f1_preditivo
        })
        
        # Gráfico de linha
        fig = px.line(df_simulacao, x="Data", y=["F1-Score (Reativo)", "F1-Score (Preditivo)"],
                     title="Simulação: Evolução do F1-Score com Diferentes Estratégias de Retreinamento")
        fig.update_layout(hovermode="x unified")
        
        # Adicionar marcadores para retreinamentos
        for i in range(12):
            if i % 3 == 0:
                fig.add_vline(x=i, line_dash="dash", line_color="green", 
                             annotation_text="Retreinamento", annotation_position="top right")
        
        st.plotly_chart(fig)
        
        st.markdown("""
        O gráfico acima simula como o F1-score poderia evoluir ao longo do tempo em duas estratégias:
        
        - **Reativa**: Sem retreinamento, o modelo degrada gradualmente
        - **Preditiva**: Com retreinamentos trimestrais, o modelo recupera seu desempenho periodicamente
        
        Na prática, a escolha da estratégia depende do equilíbrio entre custo de retreinamento e impacto da degradação de desempenho.
        """)

# Informações adicionais no rodapé
st.markdown("---")
st.markdown("**Dashboard de Monitoramento do Projeto de ML - Arremessos do Kobe Bryant**")
st.markdown("Projeto de Disciplina de Engenharia de Machine Learning - Infnet")