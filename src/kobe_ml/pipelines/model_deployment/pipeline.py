"""
Pipeline para implantação e monitoramento do modelo.
"""
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    carregar_modelo,
    preparar_dados_producao,
    aplicar_modelo,
    salvar_predicoes,
    calcular_metricas_producao,
    analisar_data_drift
)

def create_pipeline(**kwargs) -> Pipeline:
    """
    Cria o pipeline de deployment do modelo.
    
    Returns:
        Pipeline para aplicação e monitoramento do modelo em produção.
    """
    return pipeline(
        [
            node(
                func=carregar_modelo,
                inputs="modelo_final",
                outputs="modelo_carregado",
                name="carregar_modelo",
            ),
            node(
                func=preparar_dados_producao,
                inputs="kobe_prod",
                outputs="dados_prod_preparados",
                name="preparar_dados_producao",
            ),
            node(
                func=aplicar_modelo,
                inputs=["modelo_carregado", "dados_prod_preparados"],
                outputs="resultados_predicao",
                name="aplicar_modelo_producao",
            ),
            node(
                func=salvar_predicoes,
                inputs="resultados_predicao",
                outputs="predicoes",
                name="salvar_predicoes",
            ),
            node(
                func=calcular_metricas_producao,
                inputs="resultados_predicao",  # Corrigido: remove dados_prod_preparados
                outputs="metricas_producao",
                name="calcular_metricas_producao",
            ),
            node(
                func=analisar_data_drift,
                inputs=["train_set", "dados_prod_preparados"],
                outputs="analise_drift",
                name="analisar_data_drift",
            ),
        ]
    )