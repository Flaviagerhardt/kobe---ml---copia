from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    treinar_regressao_logistica,
    treinar_arvore_decisao,
    selecionar_melhor_modelo,
    aplicar_modelo
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=treinar_regressao_logistica,
            inputs=["train_set", "test_set"],
            outputs=["modelo_regressao", "metricas_regressao"],
            name="treinar_regressao_logistica"
        ),
        node(
            func=treinar_arvore_decisao,
            inputs=["train_set", "test_set"],
            outputs=["modelo_arvore", "metricas_arvore"],
            name="treinar_arvore_decisao"
        ),
        node(
            func=selecionar_melhor_modelo,
            inputs=["modelo_regressao", "metricas_regressao", "modelo_arvore", "metricas_arvore"],
            outputs=["modelo_final", "metricas_final"],
            name="selecionar_melhor_modelo"
        ),
        node(
            func=aplicar_modelo,
            inputs=["modelo_final", "kobe_prod"],
            outputs="predicoes_treino",  # Nome alterado
            name="aplicar_modelo"
        )
    ])