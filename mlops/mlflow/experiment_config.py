"""
MLflow experiment tracking configuration and utilities.
Provides comprehensive experiment management for multimodal training.
"""
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import os
import json
import torch
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

logger = logging.getLogger(__name__)


class MLflowExperimentManager:
    """
    Comprehensive MLflow experiment management for multimodal training.
    Handles experiment creation, logging, model registry, and analysis.
    """
    
    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: str = "multimodal-foundation-model",
        artifact_location: Optional[str] = None
    ):
        """
        Initialize MLflow experiment manager.
        
        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: Name of the experiment
            artifact_location: Location to store artifacts
        """
        # Set tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        elif "MLFLOW_TRACKING_URI" not in os.environ:
            # Default to local tracking
            mlflow.set_tracking_uri("file:./mlruns")
        
        self.experiment_name = experiment_name
        self.client = MlflowClient()
        
        # Create or get experiment
        try:
            self.experiment = mlflow.get_experiment_by_name(experiment_name)
            if self.experiment is None:
                experiment_id = mlflow.create_experiment(
                    experiment_name,
                    artifact_location=artifact_location
                )
                self.experiment = mlflow.get_experiment(experiment_id)
        except Exception as e:
            logger.error(f"Failed to create/get experiment: {e}")
            raise
        
        self.experiment_id = self.experiment.experiment_id
        logger.info(f"Using MLflow experiment: {experiment_name} (ID: {self.experiment_id})")
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False
    ) -> mlflow.ActiveRun:
        """
        Start MLflow run with proper configuration.
        
        Args:
            run_name: Name for the run
            tags: Additional tags for the run
            nested: Whether this is a nested run
            
        Returns:
            Active MLflow run
        """
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Default tags
        default_tags = {
            "project": "multimodal-foundation-model",
            "framework": "pytorch",
            "created_by": os.getenv("USER", "unknown")
        }
        
        if tags:
            default_tags.update(tags)
        
        run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            tags=default_tags,
            nested=nested
        )
        
        return run
    
    def log_training_config(self, config: Dict[str, Any]):
        """Log training configuration parameters."""
        # Flatten nested dictionaries
        flat_config = self._flatten_dict(config)
        
        # Log parameters
        for key, value in flat_config.items():
            if isinstance(value, (int, float, str, bool)):
                mlflow.log_param(key, value)
            else:
                mlflow.log_param(key, str(value))
        
        # Log config as artifact
        config_path = "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        mlflow.log_artifact(config_path)
        os.remove(config_path)
    
    def log_model_info(self, model: torch.nn.Module, model_name: str = "model"):
        """Log model architecture and parameter information."""
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        mlflow.log_param(f"{model_name}_total_params", total_params)
        mlflow.log_param(f"{model_name}_trainable_params", trainable_params)
        mlflow.log_param(f"{model_name}_trainable_ratio", trainable_params / total_params)
        
        # Log model size
        model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
        mlflow.log_param(f"{model_name}_size_mb", round(model_size, 2))
        
        # Log model architecture summary
        summary = str(model)
        with open(f"{model_name}_architecture.txt", 'w') as f:
            f.write(summary)
        mlflow.log_artifact(f"{model_name}_architecture.txt")
        os.remove(f"{model_name}_architecture.txt")
    
    def log_training_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        epoch: Optional[int] = None
    ):
        """Log training metrics with proper step tracking."""
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)) and not (np.isnan(value) or np.isinf(value)):
                mlflow.log_metric(metric_name, value, step=step)
        
        if epoch is not None:
            mlflow.log_metric("epoch", epoch, step=step)
    
    def log_evaluation_results(
        self,
        results: Dict[str, Any],
        dataset_name: str = "eval"
    ):
        """Log comprehensive evaluation results."""
        # Log scalar metrics
        for metric_name, value in results.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"{dataset_name}_{metric_name}", value)
        
        # Save detailed results as artifact
        results_file = f"{dataset_name}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        mlflow.log_artifact(results_file)
        os.remove(results_file)
    
    def log_model_checkpoint(
        self,
        model: torch.nn.Module,
        checkpoint_path: str,
        model_name: str = "model",
        registered_model_name: Optional[str] = None
    ):
        """Log model checkpoint and optionally register in model registry."""
        # Log model as artifact
        mlflow.pytorch.log_model(
            model,
            f"models/{model_name}",
            registered_model_name=registered_model_name
        )
        
        # Log checkpoint files
        if os.path.exists(checkpoint_path):
            mlflow.log_artifacts(checkpoint_path, artifact_path="checkpoints")
    
    def log_training_plots(
        self,
        training_history: List[Dict[str, float]],
        save_dir: str = "./plots"
    ):
        """Generate and log training visualization plots."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Convert to DataFrame
        df = pd.DataFrame(training_history)
        
        if df.empty:
            logger.warning("No training history to plot")
            return
        
        # Loss curves
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Training loss
        if 'train_loss' in df.columns:
            axes[0, 0].plot(df.index, df['train_loss'], label='Train Loss')
            if 'eval_loss' in df.columns:
                axes[0, 0].plot(df.index, df['eval_loss'], label='Validation Loss')
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # Learning rate
        if 'learning_rate' in df.columns:
            axes[0, 1].plot(df.index, df['learning_rate'])
            axes[0, 1].set_title('Learning Rate')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Learning Rate')
            axes[0, 1].grid(True)
        
        # Evaluation metrics
        eval_metrics = [col for col in df.columns if col.startswith('eval_') and col != 'eval_loss']
        if eval_metrics:
            for i, metric in enumerate(eval_metrics[:2]):  # Plot first 2 metrics
                ax = axes[1, i]
                ax.plot(df.index, df[metric])
                ax.set_title(f'Evaluation: {metric}')
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric)
                ax.grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(save_dir, "training_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log plot
        mlflow.log_artifact(plot_path)
        
        # Additional metrics plot
        if len(eval_metrics) > 2:
            fig, ax = plt.subplots(figsize=(12, 8))
            for metric in eval_metrics:
                ax.plot(df.index, df[metric], label=metric)
            ax.set_title('All Evaluation Metrics')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Score')
            ax.legend()
            ax.grid(True)
            
            plot_path = os.path.join(save_dir, "all_metrics.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            mlflow.log_artifact(plot_path)
    
    def log_dataset_info(self, dataset_info: Dict[str, Any]):
        """Log dataset information and statistics."""
        for key, value in dataset_info.items():
            if isinstance(value, (int, float, str, bool)):
                mlflow.log_param(f"dataset_{key}", value)
        
        # Save detailed dataset info
        info_file = "dataset_info.json"
        with open(info_file, 'w') as f:
            json.dump(dataset_info, f, indent=2, default=str)
        mlflow.log_artifact(info_file)
        os.remove(info_file)
    
    def log_system_info(self):
        """Log system and environment information."""
        import platform
        import psutil
        
        system_info = {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": psutil.cpu_count(),
            "memory_gb": round(psutil.virtual_memory().total / 1024**3, 2),
            "torch_version": torch.__version__,
        }
        
        # GPU information
        if torch.cuda.is_available():
            system_info.update({
                "cuda_version": torch.version.cuda,
                "gpu_count": torch.cuda.device_count(),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "Unknown"
            })
        
        for key, value in system_info.items():
            mlflow.log_param(f"system_{key}", value)
    
    def compare_runs(
        self,
        run_ids: List[str],
        metrics: List[str] = None
    ) -> pd.DataFrame:
        """Compare multiple runs and return comparison DataFrame."""
        if metrics is None:
            metrics = ["train_loss", "eval_loss", "eval_accuracy"]
        
        comparison_data = []
        
        for run_id in run_ids:
            run = self.client.get_run(run_id)
            
            run_data = {
                "run_id": run_id,
                "run_name": run.info.run_name,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time
            }
            
            # Add metrics
            for metric in metrics:
                metric_history = self.client.get_metric_history(run_id, metric)
                if metric_history:
                    run_data[metric] = metric_history[-1].value
                else:
                    run_data[metric] = None
            
            # Add key parameters
            for param_key, param_value in run.data.params.items():
                if param_key in ["learning_rate", "batch_size", "model_name"]:
                    run_data[param_key] = param_value
            
            comparison_data.append(run_data)
        
        return pd.DataFrame(comparison_data)
    
    def get_best_run(
        self,
        metric: str = "eval_loss",
        ascending: bool = True
    ) -> Optional[mlflow.entities.Run]:
        """Find the best run based on a specific metric."""
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
            max_results=1
        )
        
        if not runs.empty:
            run_id = runs.iloc[0]["run_id"]
            return self.client.get_run(run_id)
        
        return None
    
    def register_best_model(
        self,
        model_name: str,
        metric: str = "eval_loss",
        ascending: bool = True,
        stage: str = "Staging"
    ):
        """Register the best model to the model registry."""
        best_run = self.get_best_run(metric, ascending)
        
        if best_run is None:
            logger.warning("No runs found to register")
            return
        
        model_uri = f"runs:/{best_run.info.run_id}/models/model"
        
        try:
            registered_model = mlflow.register_model(model_uri, model_name)
            
            # Transition to specified stage
            self.client.transition_model_version_stage(
                name=model_name,
                version=registered_model.version,
                stage=stage
            )
            
            logger.info(f"Registered model {model_name} version {registered_model.version} in stage {stage}")
            return registered_model
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return None
    
    def _flatten_dict(self, d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Flatten nested dictionary for parameter logging."""
        flat_dict = {}
        
        for key, value in d.items():
            new_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                flat_dict.update(self._flatten_dict(value, new_key))
            else:
                flat_dict[new_key] = value
        
        return flat_dict


class MLflowCallback:
    """Callback for automatic MLflow logging during training."""
    
    def __init__(
        self,
        experiment_manager: MLflowExperimentManager,
        log_every_n_steps: int = 100,
        log_model_checkpoints: bool = True
    ):
        """
        Initialize MLflow callback.
        
        Args:
            experiment_manager: MLflow experiment manager
            log_every_n_steps: Frequency of metric logging
            log_model_checkpoints: Whether to log model checkpoints
        """
        self.experiment_manager = experiment_manager
        self.log_every_n_steps = log_every_n_steps
        self.log_model_checkpoints = log_model_checkpoints
        self.step = 0
    
    def on_train_step_end(self, trainer, logs: Dict[str, float]):
        """Called at the end of each training step."""
        self.step += 1
        
        if self.step % self.log_every_n_steps == 0:
            self.experiment_manager.log_training_metrics(logs, step=self.step)
    
    def on_epoch_end(self, trainer, epoch: int, logs: Dict[str, float]):
        """Called at the end of each epoch."""
        self.experiment_manager.log_training_metrics(logs, step=self.step, epoch=epoch)
        
        if self.log_model_checkpoints and hasattr(trainer, 'model'):
            checkpoint_path = f"checkpoint_epoch_{epoch}"
            self.experiment_manager.log_model_checkpoint(
                trainer.model,
                checkpoint_path,
                f"model_epoch_{epoch}"
            )
    
    def on_train_end(self, trainer, final_metrics: Dict[str, float]):
        """Called at the end of training."""
        self.experiment_manager.log_evaluation_results(final_metrics, "final")
        
        if hasattr(trainer, 'training_history'):
            self.experiment_manager.log_training_plots(trainer.training_history)


def create_experiment_manager(
    tracking_uri: Optional[str] = None,
    experiment_name: str = "multimodal-foundation-model"
) -> MLflowExperimentManager:
    """Factory function to create MLflow experiment manager."""
    return MLflowExperimentManager(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name
    )