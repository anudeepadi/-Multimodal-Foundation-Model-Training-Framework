"""
HuggingFace Accelerate-based distributed trainer.
Provides simple and efficient distributed training with automatic device placement.
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Callable, Union
from pathlib import Path
import json
import time
import logging
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed, ProjectConfiguration
from transformers import get_scheduler
import mlflow
from tqdm.auto import tqdm

from .distributed_trainer import TrainingConfig

logger = logging.getLogger(__name__)


class AccelerateTrainer:
    """
    Distributed trainer using HuggingFace Accelerate.
    
    Provides simplified distributed training with automatic device placement,
    gradient synchronization, and mixed precision support.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataloader: torch.utils.data.DataLoader,
        eval_dataloader: Optional[torch.utils.data.DataLoader] = None,
        compute_metrics: Optional[Callable] = None,
        accelerate_config: Optional[str] = None
    ):
        """
        Initialize Accelerate trainer.
        
        Args:
            model: Model to train
            config: Training configuration
            train_dataloader: Training data loader
            eval_dataloader: Evaluation data loader
            compute_metrics: Function to compute evaluation metrics
            accelerate_config: Path to accelerate config file
        """
        self.config = config
        self.compute_metrics = compute_metrics
        
        # Set reproducibility
        set_seed(config.seed)
        
        # Initialize Accelerator
        kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        
        project_config = ProjectConfiguration(
            project_dir=config.output_dir,
            automatic_checkpoint_naming=True,
            total_limit=config.save_total_limit
        )
        
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision="bf16" if config.bf16 else ("fp16" if config.fp16 else "no"),
            log_with=["mlflow"] if config.use_mlflow else None,
            project_config=project_config,
            kwargs_handlers=[kwargs]
        )
        
        # Set up model and optimization
        self.model = model
        self._setup_optimizer()
        self._setup_scheduler()
        
        # Prepare everything with accelerator
        (
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.eval_dataloader,
            self.lr_scheduler
        ) = self.accelerator.prepare(
            model, self.optimizer, train_dataloader, eval_dataloader, self.lr_scheduler
        )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = None
        self.training_history = []
        
        # Set up logging
        self._setup_logging()
        
        # Print training info
        if self.accelerator.is_main_process:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,}")
            logger.info(f"Training on {self.accelerator.num_processes} devices")
    
    def _setup_optimizer(self):
        """Set up optimizer with parameter groups."""
        # Separate parameters for weight decay
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "ln_"]
        
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8
        )
    
    def _setup_scheduler(self):
        """Set up learning rate scheduler."""
        # Calculate total training steps
        num_update_steps_per_epoch = len(self.train_dataloader) // self.config.gradient_accumulation_steps
        max_train_steps = self.config.num_train_epochs * num_update_steps_per_epoch
        
        if self.config.max_steps > 0:
            max_train_steps = min(max_train_steps, self.config.max_steps)
            self.config.num_train_epochs = max_train_steps // num_update_steps_per_epoch + 1
        
        num_warmup_steps = int(max_train_steps * self.config.warmup_ratio)
        
        self.lr_scheduler = get_scheduler(
            name=self.config.scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=max_train_steps
        )
        
        self.max_train_steps = max_train_steps
        self.num_warmup_steps = num_warmup_steps
    
    def _setup_logging(self):
        """Set up experiment tracking."""
        if self.config.use_mlflow and self.accelerator.is_main_process:
            # Initialize tracking
            init_kwargs = {
                "mlflow": {
                    "experiment_name": self.config.experiment_name,
                    "run_name": self.config.run_name or f"accelerate_run_{int(time.time())}"
                }
            }
            self.accelerator.init_trackers("multimodal-training", config=self.config.to_dict(), init_kwargs=init_kwargs)
    
    def train(self) -> Dict[str, Any]:
        """
        Main training loop.
        
        Returns:
            Dictionary containing training metrics and history
        """
        logger.info(f"Starting training for {self.config.num_train_epochs} epochs")
        logger.info(f"Total training steps: {self.max_train_steps}")
        logger.info(f"Warmup steps: {self.num_warmup_steps}")
        
        # Create output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        if self.accelerator.is_main_process:
            self.config.save(output_dir / "training_config.json")
        
        # Training loop
        progress_bar = tqdm(
            range(self.max_train_steps),
            disable=not self.accelerator.is_local_main_process,
            desc="Training"
        )
        
        for epoch in range(self.config.num_train_epochs):
            self.epoch = epoch
            
            # Training epoch
            train_metrics = self._train_epoch(progress_bar)
            
            # Evaluation
            eval_metrics = {}
            if self.eval_dataloader is not None:
                eval_metrics = self._eval_epoch()
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **eval_metrics}
            epoch_metrics['epoch'] = epoch
            epoch_metrics['learning_rate'] = self.lr_scheduler.get_last_lr()[0]
            
            self.training_history.append(epoch_metrics)
            
            # Logging
            if self.accelerator.is_main_process:
                logger.info(f"Epoch {epoch}: {epoch_metrics}")
                
                # Log to trackers
                self.accelerator.log(epoch_metrics, step=epoch)
            
            # Checkpointing
            if (epoch + 1) % max(1, self.config.save_steps // len(self.train_dataloader)) == 0:
                self._save_checkpoint(epoch_metrics)
            
            # Early stopping check
            if self.global_step >= self.max_train_steps:
                break
        
        progress_bar.close()
        
        # Final evaluation
        final_metrics = {}
        if self.eval_dataloader is not None:
            final_metrics = self._eval_epoch()
            logger.info(f"Final evaluation: {final_metrics}")
            
            if self.accelerator.is_main_process:
                final_log = {f"final_{k}": v for k, v in final_metrics.items()}
                self.accelerator.log(final_log)
        
        # End experiment tracking
        if self.config.use_mlflow:
            self.accelerator.end_training()
        
        return {
            'training_history': self.training_history,
            'final_metrics': final_metrics,
            'total_steps': self.global_step
        }
    
    def _train_epoch(self, progress_bar) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_steps = 0
        
        for step, batch in enumerate(self.train_dataloader):
            with self.accelerator.accumulate(self.model):
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs
                
                # Backward pass
                self.accelerator.backward(loss)
                
                # Gradient clipping
                if self.config.gradient_clipping > 0:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.gradient_clipping)
                
                # Optimizer step
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
            
            # Update progress
            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.global_step += 1
                num_steps += 1
                
                # Gather loss across all processes
                loss_gathered = self.accelerator.gather(loss.repeat(batch[list(batch.keys())[0]].shape[0]))
                total_loss += loss_gathered.mean().item()
                
                # Periodic logging
                if self.global_step % self.config.logging_steps == 0:
                    avg_loss = total_loss / num_steps
                    current_lr = self.lr_scheduler.get_last_lr()[0]
                    
                    log_data = {
                        "step": self.global_step,
                        "train_loss": avg_loss,
                        "learning_rate": current_lr
                    }
                    
                    if self.accelerator.is_main_process:
                        logger.info(f"Step {self.global_step}: loss={avg_loss:.4f}, lr={current_lr:.2e}")
                        self.accelerator.log(log_data, step=self.global_step)
                
                # Check if we've reached max steps
                if self.global_step >= self.max_train_steps:
                    break
        
        avg_loss = total_loss / max(num_steps, 1)
        return {'train_loss': avg_loss}
    
    def _eval_epoch(self) -> Dict[str, float]:
        """Evaluate model for one epoch."""
        if self.eval_dataloader is None:
            return {}
        
        self.model.eval()
        total_loss = 0
        num_steps = 0
        all_predictions = []
        all_labels = []
        
        eval_progress = tqdm(
            self.eval_dataloader,
            desc="Evaluating",
            disable=not self.accelerator.is_local_main_process
        )
        
        with torch.no_grad():
            for batch in eval_progress:
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs
                
                # Gather losses across all processes
                losses = self.accelerator.gather_for_metrics(loss.repeat(batch[list(batch.keys())[0]].shape[0]))
                total_loss += losses.mean().item()
                num_steps += 1
                
                # Collect predictions for metrics computation
                if self.compute_metrics and hasattr(outputs, 'logits'):
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    
                    # Gather predictions and labels
                    predictions = self.accelerator.gather_for_metrics(predictions)
                    labels = self.accelerator.gather_for_metrics(batch.get('labels'))
                    
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
        
        metrics = {'eval_loss': total_loss / max(num_steps, 1)}
        
        # Compute additional metrics
        if self.compute_metrics and all_predictions and self.accelerator.is_main_process:
            additional_metrics = self.compute_metrics(all_predictions, all_labels)
            metrics.update(additional_metrics)
        
        # Broadcast metrics to all processes
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                tensor_value = torch.tensor(value, device=self.accelerator.device)
                gathered_value = self.accelerator.gather(tensor_value)
                metrics[key] = gathered_value.mean().item()
        
        return metrics
    
    def _save_checkpoint(self, metrics: Dict[str, float]):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.output_dir) / f"checkpoint-{self.global_step}"
        
        # Save model using accelerator
        self.accelerator.save_state(checkpoint_dir)
        
        # Save additional training state
        if self.accelerator.is_main_process:
            training_state = {
                'epoch': self.epoch,
                'global_step': self.global_step,
                'best_metric': self.best_metric,
                'training_history': self.training_history,
                'metrics': metrics,
                'max_train_steps': self.max_train_steps
            }
            
            with open(checkpoint_dir / "training_state.json", 'w') as f:
                json.dump(training_state, f, indent=2)
            
            logger.info(f"Checkpoint saved to {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        self.accelerator.load_state(checkpoint_path)
        
        # Load training state
        training_state_path = Path(checkpoint_path) / "training_state.json"
        if training_state_path.exists():
            with open(training_state_path, 'r') as f:
                training_state = json.load(f)
            
            self.epoch = training_state.get('epoch', 0)
            self.global_step = training_state.get('global_step', 0)
            self.best_metric = training_state.get('best_metric')
            self.training_history = training_state.get('training_history', [])
            
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def save_model(self, output_dir: str):
        """Save final model."""
        self.accelerator.wait_for_everyone()
        
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        
        # Save using accelerator's save method
        self.accelerator.save_model(unwrapped_model, output_dir)
        
        if self.accelerator.is_main_process:
            logger.info(f"Model saved to {output_dir}")
    
    def push_to_hub(
        self,
        repo_id: str,
        commit_message: str = "Upload model",
        private: bool = False,
        token: Optional[str] = None
    ):
        """Push model to HuggingFace Hub."""
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        
        if self.accelerator.is_main_process:
            unwrapped_model.push_to_hub(
                repo_id=repo_id,
                commit_message=commit_message,
                private=private,
                token=token
            )
            logger.info(f"Model pushed to hub: {repo_id}")