"""
LLaVA (Large Language and Vision Assistant) fine-tuning implementation.
Supports multimodal conversation and visual question answering tasks.
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union
from transformers import (
    LlavaForConditionalGeneration, 
    LlavaProcessor, 
    LlamaTokenizer,
    AutoProcessor
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class LLaVAFineTuner(nn.Module):
    """
    LLaVA model fine-tuning with LoRA/QLoRA support.
    
    Supports both full fine-tuning and parameter-efficient methods for
    visual instruction tuning and multimodal conversation tasks.
    """
    
    def __init__(
        self,
        model_name: str = "llava-hf/llava-1.5-7b-hf",
        lora_r: int = 64,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,
        use_qlora: bool = False,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16
    ):
        """
        Initialize LLaVA model for fine-tuning.
        
        Args:
            model_name: HuggingFace model identifier
            lora_r: LoRA rank parameter
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout rate
            target_modules: Modules to apply LoRA to
            use_qlora: Use QLoRA (4-bit quantized LoRA)
            load_in_4bit: Load model in 4-bit precision
            load_in_8bit: Load model in 8-bit precision
            device_map: Device mapping strategy
            torch_dtype: Model data type
        """
        super().__init__()
        self.model_name = model_name
        self.use_qlora = use_qlora
        
        # Quantization configuration
        quantization_config = None
        if load_in_4bit or use_qlora:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif load_in_8bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # Load base model
        self.base_model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # Configure target modules for LoRA
        if target_modules is None:
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        
        # Apply LoRA if specified
        if lora_r > 0:
            # Prepare model for k-bit training if using quantization
            if quantization_config is not None:
                self.base_model = prepare_model_for_kbit_training(self.base_model)
            
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            
            self.model = get_peft_model(self.base_model, lora_config)
            logger.info(f"Applied LoRA with rank {lora_r}")
        else:
            self.model = self.base_model
            logger.info("Using full fine-tuning")
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        self.print_trainable_parameters()
    
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
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through LLaVA model.
        
        Args:
            input_ids: Tokenized text input
            attention_mask: Attention mask
            pixel_values: Image tensor
            labels: Target labels for loss computation
            
        Returns:
            Model outputs including loss and logits
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            **kwargs
        )
        return outputs
    
    def generate_response(
        self,
        image: Union[Image.Image, str],
        prompt: str,
        max_new_tokens: int = 256,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate response for image and text prompt.
        
        Args:
            image: PIL Image or path to image
            prompt: Text prompt/question
            max_new_tokens: Maximum tokens to generate
            do_sample: Whether to use sampling
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated response text
        """
        # Load image if path provided
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # Prepare inputs
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
        
        # Decode response
        generated_text = self.processor.batch_decode(
            generated_ids[:, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )[0].strip()
        
        return generated_text
    
    def prepare_training_data(
        self,
        conversations: List[Dict[str, Any]],
        images: List[Image.Image],
        max_length: int = 2048
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare training data for instruction tuning.
        
        Args:
            conversations: List of conversation dictionaries
            images: List of PIL Images
            max_length: Maximum sequence length
            
        Returns:
            Processed training batch
        """
        batch_inputs = []
        batch_labels = []
        
        for conv, image in zip(conversations, images):
            # Format conversation
            formatted_conv = self.format_conversation(conv)
            
            # Process inputs
            inputs = self.processor(
                images=image,
                text=formatted_conv,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True
            )
            
            batch_inputs.append(inputs)
            
            # Create labels (mask input tokens, only compute loss on response)
            labels = inputs['input_ids'].clone()
            # Implementation depends on conversation format
            batch_labels.append(labels)
        
        # Collate batch
        collated_batch = self.collate_batch(batch_inputs, batch_labels)
        return collated_batch
    
    def format_conversation(self, conversation: Dict[str, Any]) -> str:
        """Format conversation for training."""
        # Standard LLaVA conversation format
        formatted = ""
        for turn in conversation.get('conversations', []):
            role = turn.get('from', 'human')
            content = turn.get('value', '')
            
            if role == 'human':
                formatted += f"USER: {content}\n"
            elif role == 'gpt':
                formatted += f"ASSISTANT: {content}\n"
        
        return formatted
    
    def collate_batch(self, inputs_list, labels_list):
        """Collate batch of inputs and labels."""
        # Implementation for proper batching
        batch = {}
        
        # Stack tensors
        if inputs_list:
            keys = inputs_list[0].keys()
            for key in keys:
                batch[key] = torch.stack([inp[key].squeeze() for inp in inputs_list])
            
            batch['labels'] = torch.stack([lbl.squeeze() for lbl in labels_list])
        
        return batch
    
    def save_pretrained(self, save_path: str):
        """Save model weights."""
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(save_path)
        else:
            self.base_model.save_pretrained(save_path)
        
        self.processor.save_pretrained(save_path)
        logger.info(f"Model saved to {save_path}")
    
    def load_adapter(self, adapter_path: str):
        """Load LoRA adapter weights."""
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.base_model, adapter_path)
        logger.info(f"Adapter loaded from {adapter_path}")
    
    def merge_and_save(self, save_path: str):
        """Merge LoRA weights and save full model."""
        if hasattr(self.model, 'merge_and_unload'):
            merged_model = self.model.merge_and_unload()
            merged_model.save_pretrained(save_path)
            self.processor.save_pretrained(save_path)
            logger.info(f"Merged model saved to {save_path}")


class LLaVADataCollator:
    """Data collator for LLaVA training."""
    
    def __init__(self, processor, max_length: int = 2048):
        self.processor = processor
        self.max_length = max_length
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch of training examples."""
        images = [item['image'] for item in batch]
        texts = [item['text'] for item in batch]
        
        # Process batch
        processed = self.processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        # Create labels
        labels = processed['input_ids'].clone()
        # Mask input tokens based on conversation format
        # This needs to be implemented based on specific format
        
        processed['labels'] = labels
        return processed


def create_llava_model(config: Dict[str, Any]) -> LLaVAFineTuner:
    """Factory function to create LLaVA model from config."""
    return LLaVAFineTuner(
        model_name=config.get("model_name", "llava-hf/llava-1.5-7b-hf"),
        lora_r=config.get("lora_r", 64),
        lora_alpha=config.get("lora_alpha", 16),
        lora_dropout=config.get("lora_dropout", 0.1),
        target_modules=config.get("target_modules"),
        use_qlora=config.get("use_qlora", False),
        load_in_4bit=config.get("load_in_4bit", False),
        load_in_8bit=config.get("load_in_8bit", False),
        device_map=config.get("device_map", "auto"),
        torch_dtype=getattr(torch, config.get("torch_dtype", "bfloat16"))
    )