"""
CLIP Model with LoRA fine-tuning implementation.
Supports efficient parameter adaptation for vision-language tasks.
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from transformers import CLIPModel, CLIPProcessor, CLIPConfig
from peft import LoraConfig, get_peft_model, TaskType
import logging

logger = logging.getLogger(__name__)


class CLIPLoRAModel(nn.Module):
    """
    CLIP model with LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.
    
    This implementation allows fine-tuning of CLIP models with minimal parameter overhead
    while maintaining high performance on vision-language tasks.
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: Optional[list] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize CLIP model with LoRA adaptation.
        
        Args:
            model_name: HuggingFace model identifier
            lora_r: LoRA rank parameter
            lora_alpha: LoRA alpha scaling parameter
            lora_dropout: Dropout rate for LoRA layers
            target_modules: List of modules to apply LoRA to
            device: Device to load model on
        """
        super().__init__()
        self.model_name = model_name
        self.device = device
        
        # Load base CLIP model
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Default target modules for CLIP LoRA
        if target_modules is None:
            target_modules = [
                "q_proj", "v_proj", "k_proj", "out_proj",
                "fc1", "fc2"
            ]
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.clip_model, lora_config)
        self.model.to(device)
        
        logger.info(f"Initialized CLIP LoRA model with {self.get_trainable_params()} trainable parameters")
    
    def get_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def forward(
        self, 
        input_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_loss: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through CLIP model.
        
        Args:
            input_ids: Tokenized text input
            pixel_values: Image tensor
            attention_mask: Attention mask for text
            return_loss: Whether to compute contrastive loss
            
        Returns:
            Dictionary containing logits and optional loss
        """
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            return_loss=return_loss,
            **kwargs
        )
        return outputs
    
    def encode_text(self, text: list) -> torch.Tensor:
        """Encode text inputs to embeddings."""
        inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        
        return text_features
    
    def encode_image(self, images: list) -> torch.Tensor:
        """Encode image inputs to embeddings."""
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        
        return image_features
    
    def compute_similarity(self, text: list, images: list) -> torch.Tensor:
        """Compute cosine similarity between text and images."""
        text_features = self.encode_text(text)
        image_features = self.encode_image(images)
        
        # Normalize features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Compute similarity
        similarity = torch.matmul(text_features, image_features.T)
        return similarity
    
    def save_pretrained(self, save_path: str):
        """Save LoRA model weights."""
        self.model.save_pretrained(save_path)
        logger.info(f"Model saved to {save_path}")
    
    def load_pretrained(self, load_path: str):
        """Load LoRA model weights."""
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.clip_model, load_path)
        self.model.to(self.device)
        logger.info(f"Model loaded from {load_path}")
    
    def merge_and_unload(self):
        """Merge LoRA weights into base model and return unloaded model."""
        merged_model = self.model.merge_and_unload()
        return merged_model
    
    def print_trainable_parameters(self):
        """Print statistics about trainable parameters."""
        trainable_params = 0
        all_param = 0
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        print(f"Trainable params: {trainable_params:,} || "
              f"All params: {all_param:,} || "
              f"Trainable%: {100 * trainable_params / all_param:.4f}")


class CLIPContrastiveLoss(nn.Module):
    """Contrastive loss for CLIP training."""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(self, text_embeddings: torch.Tensor, image_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss between text and image embeddings.
        
        Args:
            text_embeddings: Normalized text embeddings [batch_size, embed_dim]
            image_embeddings: Normalized image embeddings [batch_size, embed_dim]
            
        Returns:
            Contrastive loss value
        """
        batch_size = text_embeddings.shape[0]
        
        # Compute similarity matrix
        logits = torch.matmul(text_embeddings, image_embeddings.T) / self.temperature
        
        # Labels are diagonal (i-th text matches i-th image)
        labels = torch.arange(batch_size, device=logits.device)
        
        # Symmetric loss (text-to-image + image-to-text)
        loss_t2i = self.cross_entropy(logits, labels)
        loss_i2t = self.cross_entropy(logits.T, labels)
        
        return (loss_t2i + loss_i2t) / 2


def create_clip_lora_model(config: Dict[str, Any]) -> CLIPLoRAModel:
    """Factory function to create CLIP LoRA model from config."""
    return CLIPLoRAModel(
        model_name=config.get("model_name", "openai/clip-vit-base-patch32"),
        lora_r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.1),
        target_modules=config.get("target_modules"),
        device=config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )