import mlflow
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ExperimentTracker:
    def __init__(self, experiment_name: str = "ShopSense-Recommendations", tracking_uri: str = "sqlite:///mlflow.db"):
        self.experiment_name = experiment_name
        
        # Set up MLflow
        mlflow.set_tracking_uri(tracking_uri)
        try:
            self.experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"Created new MLflow experiment: {experiment_name}")
        except mlflow.exceptions.MlflowException:
            self.experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
            logger.info(f"Using existing MLflow experiment: {experiment_name}")
            
    def start_run(self, run_name: str):
        mlflow.set_experiment(self.experiment_name)
        return mlflow.start_run(run_name=run_name)
        
    def log_params(self, params: Dict[str, Any]):
        mlflow.log_params(params)
        
    def log_metrics(self, metrics: Dict[str, float]):
        mlflow.log_metrics(metrics)
        
    def log_artifact(self, local_path: str, artifact_path: str = None):
        mlflow.log_artifact(local_path, artifact_path)
        
    def end_run(self):
        mlflow.end_run()
