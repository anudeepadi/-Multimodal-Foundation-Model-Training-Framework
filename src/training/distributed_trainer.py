"""
Distributed training implementation with support for multiple backends.
Provides unified interface for DeepSpeed, FSDP, and standard distributed training.
"""
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union, Callable
from pathlib import Path
import json
import time
import logging
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import mlflow

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration class for distributed training."""
    
    # Model and data
    model_name: str = "openai/clip-vit-base-patch32"
    dataset_name: str = "ms_coco"
    max_seq_length: int = 2048
    
    # Training hyperparameters
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    num_train_epochs: int = 3
    max_steps: int = -1
    
    # Distributed training
    distributed_backend: str = "nccl"  # nccl, gloo, mpi
    gradient_clipping: float = 1.0
    fp16: bool = False
    bf16: bool = True
    
    # FSDP configuration
    use_fsdp: bool = False
    fsdp_transformer_layer_cls: Optional[str] = None
    fsdp_min_num_params: int = 1e8
    cpu_offload: bool = False
    
    # DeepSpeed configuration
    use_deepspeed: bool = False
    deepspeed_config: Optional[str] = None
    
    # Checkpointing and logging
    output_dir: str = "./outputs"
    logging_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 1000
    save_total_limit: int = 3
    
    # Optimization
    scheduler_type: str = "cosine"  # linear, cosine, polynomial
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    
    # Monitoring
    use_mlflow: bool = True
    experiment_name: str = "multimodal-training"
    run_name: Optional[str] = None
    
    # Reproducibility
    seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def save(self, path: str):
        """Save configuration to file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        """Load configuration from file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class DistributedTrainer:
    """
    Distributed trainer supporting multiple backends and optimization strategies.
    
    Features:
    - Multi-GPU training with DDP, FSDP, or DeepSpeed
    - Mixed precision training (FP16/BF16)
    - Gradient accumulation and clipping
    - Learning rate scheduling
    - Checkpointing and resuming
    - Comprehensive logging and monitoring
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: TrainingConfig,
        train_dataloader: torch.utils.data.DataLoader,
        eval_dataloader: Optional[torch.utils.data.DataLoader] = None,
        compute_metrics: Optional[Callable] = None
    ):
        """
        Initialize distributed trainer.
        
        Args:
            model: Model to train
            config: Training configuration
            train_dataloader: Training data loader
            eval_dataloader: Evaluation data loader
            compute_metrics: Function to compute evaluation metrics
        """
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.compute_metrics = compute_metrics
        
        # Set up distributed training
        self._setup_distributed()
        self._setup_model()
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_logging()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = None
        self.training_history = []
        
    def _setup_distributed(self):
        """Initialize distributed training environment."""
        if not dist.is_initialized():
            if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
                dist.init_process_group(
                    backend=self.config.distributed_backend,
                    init_method='env://'
                )
            else:
                logger.info("Running in single-GPU mode")
                return
        
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # Set device
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f'cuda:{self.local_rank}')
        else:
            self.device = torch.device('cpu')
        
        logger.info(f"Rank: {self.rank}, World size: {self.world_size}, Device: {self.device}")
    
    def _setup_model(self):
        """Set up model for distributed training."""
        self.model.to(self.device)
        
        if self.config.use_deepspeed:
            self._setup_deepspeed()
        elif self.config.use_fsdp and self.world_size > 1:
            self._setup_fsdp()
        elif self.world_size > 1:
            self._setup_ddp()
        
        # Enable gradient checkpointing if supported
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
    
    def _setup_ddp(self):
        """Set up DistributedDataParallel."""
        self.model = DDP(
            self.model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=False
        )
        logger.info("Initialized DDP")
    
    def _setup_fsdp(self):
        """Set up Fully Sharded Data Parallel."""
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
        from transformers.models.clip.modeling_clip import CLIPEncoderLayer
        
        # Determine transformer layer class
        transformer_cls = None
        if self.config.fsdp_transformer_layer_cls:
            # Import the specified class
            module_name, class_name = self.config.fsdp_transformer_layer_cls.rsplit('.', 1)
            module = __import__(module_name, fromlist=[class_name])
            transformer_cls = getattr(module, class_name)
        else:
            # Try to detect automatically
            if hasattr(self.model, 'text_model'):  # CLIP-like
                transformer_cls = CLIPEncoderLayer
            elif hasattr(self.model, 'language_model'):  # LLaVA-like
                transformer_cls = LlamaDecoderLayer
        
        # Auto wrap policy
        auto_wrap_policy = None
        if transformer_cls:
            auto_wrap_policy = transformer_auto_wrap_policy.partial(
                transformer_layer_cls={transformer_cls}
            )
        
        # CPU offload configuration
        cpu_offload = CPUOffload(offload_params=True) if self.config.cpu_offload else None
        
        self.model = FSDP(
            self.model,
            auto_wrap_policy=auto_wrap_policy,
            cpu_offload=cpu_offload,
            mixed_precision=self._get_fsdp_mixed_precision(),
            sharding_strategy=FSDP.ShardingStrategy.FULL_SHARD,
            device_id=self.local_rank,
            limit_all_gathers=True,
            use_orig_params=True
        )
        logger.info("Initialized FSDP")
    
    def _get_fsdp_mixed_precision(self):
        """Get FSDP mixed precision policy."""
        from torch.distributed.fsdp import MixedPrecision
        
        if self.config.bf16:
            return MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16
            )
        elif self.config.fp16:
            return MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16
            )
        return None
    
    def _setup_deepspeed(self):
        """Set up DeepSpeed training."""
        try:
            import deepspeed
        except ImportError:
            raise ImportError("DeepSpeed not installed. Install with: pip install deepspeed")
        
        if self.config.deepspeed_config:
            with open(self.config.deepspeed_config, 'r') as f:
                ds_config = json.load(f)
        else:
            ds_config = self._get_default_deepspeed_config()
        
        self.model, self.optimizer, _, self.lr_scheduler = deepspeed.initialize(
            model=self.model,
            config=ds_config,
            model_parameters=self.model.parameters()
        )
        
        self._deepspeed_engine = self.model
        logger.info("Initialized DeepSpeed")
    
    def _get_default_deepspeed_config(self) -> Dict[str, Any]:
        """Get default DeepSpeed configuration."""
        return {
            "train_batch_size": self.config.batch_size * self.config.gradient_accumulation_steps * self.world_size,
            "train_micro_batch_size_per_gpu": self.config.batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": self.config.learning_rate,
                    "weight_decay": self.config.weight_decay,
                    "beta1": 0.9,
                    "beta2": 0.95
                }
            },
            "scheduler": {
                "type": "WarmupCosineLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": self.config.learning_rate,
                    "warmup_num_steps": int(len(self.train_dataloader) * self.config.warmup_ratio),
                    "total_num_steps": len(self.train_dataloader) * self.config.num_train_epochs
                }
            },
            "fp16": {
                "enabled": self.config.fp16,
                "auto_cast": False,
                "loss_scale": 0,
                "initial_scale_power": 16,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1
            },
            "bf16": {
                "enabled": self.config.bf16
            },
            "zero_optimization": {
                "stage": 2,
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e8,
                "contiguous_gradients": True
            },
            "gradient_clipping": self.config.gradient_clipping,
            "wall_clock_breakdown": False
        }
    
    def _setup_optimizer(self):
        """Set up optimizer."""
        if self.config.use_deepspeed:
            # Optimizer handled by DeepSpeed
            return
        
        # Separate parameters for weight decay
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "ln_"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay) and p.requires_grad],
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
        if self.config.use_deepspeed:
            # Scheduler handled by DeepSpeed
            return
        
        num_training_steps = len(self.train_dataloader) * self.config.num_train_epochs
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        
        if self.config.scheduler_type == "linear":
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        elif self.config.scheduler_type == "cosine":
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        else:
            self.lr_scheduler = None
    
    def _setup_logging(self):
        """Set up logging and experiment tracking."""
        if self.config.use_mlflow and self.rank == 0:
            mlflow.set_experiment(self.config.experiment_name)
            run_name = self.config.run_name or f"run_{int(time.time())}"
            mlflow.start_run(run_name=run_name)
            mlflow.log_params(self.config.to_dict())
    
    def train(self) -> Dict[str, Any]:
        """
        Main training loop.
        
        Returns:
            Dictionary containing training metrics and history
        """
        logger.info(f"Starting training for {self.config.num_train_epochs} epochs")
        
        # Create output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        if self.rank == 0:
            self.config.save(output_dir / "training_config.json")
        
        total_steps = len(self.train_dataloader) * self.config.num_train_epochs
        
        for epoch in range(self.config.num_train_epochs):
            self.epoch = epoch
            
            # Set epoch for distributed sampler
            if hasattr(self.train_dataloader.sampler, 'set_epoch'):
                self.train_dataloader.sampler.set_epoch(epoch)
            
            # Training epoch
            train_metrics = self._train_epoch()
            
            # Evaluation
            eval_metrics = {}
            if self.eval_dataloader is not None:
                eval_metrics = self._eval_epoch()
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **eval_metrics}
            epoch_metrics['epoch'] = epoch
            self.training_history.append(epoch_metrics)
            
            # Logging
            if self.rank == 0:
                logger.info(f"Epoch {epoch}: {epoch_metrics}")
                if self.config.use_mlflow:
                    mlflow.log_metrics(epoch_metrics, step=epoch)
            
            # Checkpointing
            if (epoch + 1) % (self.config.save_steps // len(self.train_dataloader)) == 0:
                self._save_checkpoint(epoch_metrics)
        
        # Final evaluation
        if self.eval_dataloader is not None:
            final_metrics = self._eval_epoch()
            logger.info(f"Final evaluation: {final_metrics}")
            if self.rank == 0 and self.config.use_mlflow:
                mlflow.log_metrics({f"final_{k}": v for k, v in final_metrics.items()})
        
        # End MLflow run
        if self.config.use_mlflow and self.rank == 0:
            mlflow.end_run()
        
        return {
            'training_history': self.training_history,
            'final_metrics': final_metrics if 'final_metrics' in locals() else {},
            'total_steps': self.global_step
        }
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_steps = 0
        
        for step, batch in enumerate(self.train_dataloader):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=self.config.fp16 or self.config.bf16):
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs
                
                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.config.use_deepspeed:
                self._deepspeed_engine.backward(loss)
            else:
                loss.backward()
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            
            # Optimizer step
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                if self.config.use_deepspeed:
                    self._deepspeed_engine.step()
                else:
                    # Gradient clipping
                    if self.config.gradient_clipping > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.config.gradient_clipping
                        )
                    
                    self.optimizer.step()
                    if self.lr_scheduler:
                        self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                
                self.global_step += 1
                num_steps += 1
                
                # Logging
                if self.global_step % self.config.logging_steps == 0 and self.rank == 0:
                    current_lr = self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler else self.config.learning_rate
                    logger.info(f"Step {self.global_step}: loss={loss.item():.4f}, lr={current_lr:.2e}")
        
        avg_loss = total_loss / num_steps if num_steps > 0 else 0
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
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs
                
                total_loss += loss.item()
                num_steps += 1
                
                # Collect predictions for metrics computation
                if self.compute_metrics and hasattr(outputs, 'logits'):
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(batch.get('labels', []).cpu().numpy())
        
        metrics = {'eval_loss': total_loss / num_steps if num_steps > 0 else 0}
        
        # Compute additional metrics
        if self.compute_metrics and all_predictions:
            additional_metrics = self.compute_metrics(all_predictions, all_labels)
            metrics.update(additional_metrics)
        
        return metrics
    
    def _save_checkpoint(self, metrics: Dict[str, float]):
        """Save model checkpoint."""
        if self.rank != 0:
            return
        
        checkpoint_dir = Path(self.config.output_dir) / f"checkpoint-{self.global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        if self.config.use_deepspeed:
            self._deepspeed_engine.save_checkpoint(str(checkpoint_dir))
        elif self.config.use_fsdp:
            # FSDP checkpoint saving
            from torch.distributed.fsdp import StateDictType, FullStateDictConfig
            
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
                state_dict = self.model.state_dict()
            
            if self.rank == 0:
                torch.save(state_dict, checkpoint_dir / "pytorch_model.bin")
        else:
            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
            torch.save(model_to_save.state_dict(), checkpoint_dir / "pytorch_model.bin")
        
        # Save training state
        training_state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_metric': self.best_metric,
            'training_history': self.training_history,
            'metrics': metrics
        }
        
        with open(checkpoint_dir / "training_state.json", 'w') as f:
            json.dump(training_state, f, indent=2)
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
    
    def cleanup(self):
        """Clean up distributed training."""
        if dist.is_initialized():
            dist.destroy_process_group()