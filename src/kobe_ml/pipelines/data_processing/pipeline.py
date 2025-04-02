from kedro.pipeline import Pipeline, node, pipeline
from .nodes import preparar_dados, splitar_dados

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=preparar_dados,
            inputs="kobe_dev",  # Continue usando o original
            outputs="dados_filtrados",
            name="preparar_dados"
        ),
        node(
            func=splitar_dados,
            inputs=["dados_filtrados", "params:test_size"],
            outputs=["train_set", "test_set"],
            name="splitar_dados"
        )
    ])