from kedro_mlflow.framework.hooks import MlflowHook
from kedro.framework.hooks import hook_impl
from kedro.framework.project import settings

class KobeMLflowHook(MlflowHook):
    @hook_impl
    def after_catalog_created(self, catalog):
        """Hook to be called after the catalog is created."""
        # Verificar se estamos no pipeline de processamento de dados
        if "kobe_dev" in catalog._data_sets:
            self._log_dataset_metrics(catalog)
    
    def _log_dataset_metrics(self, catalog):
        """Log de métricas dos datasets no MLflow."""
        import mlflow
        
        try:
            # Acessar datasets
            kobe_dev = catalog._get_dataset("kobe_dev")
            data = kobe_dev()
            
            # Registrar métricas
            with mlflow.start_run(run_name="MetricasDatasets"):
                mlflow.log_metric("total_registros", len(data))
                mlflow.log_metric("colunas", data.shape[1])
                
                # Registrar distribuição da variável alvo
                if "shot_made_flag" in data.columns:
                    mlflow.log_metric("acertos", data["shot_made_flag"].sum())
                    mlflow.log_metric("erros", len(data) - data["shot_made_flag"].sum())
                    mlflow.log_metric("taxa_acerto", data["shot_made_flag"].mean())
        except Exception as e:
            print(f"Erro ao registrar métricas dos datasets: {e}")