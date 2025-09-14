"""
Multimodal LoRA Utilities for HuggingFace PEFT Library

This module provides specialized utilities for configuring LoRA on 
multimodal models including vision-language models like CLIP, BLIP, and LLaVA.

Author: Contributing to HuggingFace PEFT
License: Apache-2.0 (matching PEFT library)
"""
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from peft import LoraConfig, TaskType
import logging

logger = logging.getLogger(__name__)


@dataclass
class MultimodalLoRAConfig:
    """Configuration for multimodal LoRA setup with separate vision and text configs."""
    
    vision_config: Optional[LoraConfig] = None
    text_config: Optional[LoraConfig] = None
    shared_config: Optional[LoraConfig] = None
    fusion_modules: Optional[List[str]] = None


def get_vision_lora_config(
    architecture: str = "clip",
    rank: int = 16,
    alpha: Optional[int] = None,
    dropout: float = 0.1,
    **kwargs
) -> LoraConfig:
    """
    Get optimized LoRA configuration for vision encoders.
    
    Args:
        architecture: Vision architecture type ('clip', 'vit', 'swin', 'resnet')
        rank: LoRA rank
        alpha: LoRA alpha (defaults to 2*rank)
        dropout: LoRA dropout rate
        **kwargs: Additional LoRA parameters
        
    Returns:
        LoraConfig optimized for vision models
    """
    if alpha is None:
        alpha = rank * 2
    
    # Architecture-specific target modules
    if architecture.lower() in ['clip', 'clip-vit']:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "out_proj",  # Self-attention
            "fc1", "fc2",  # MLP layers
        ]
    elif architecture.lower() in ['vit', 'deit']:
        target_modules = [
            "query", "key", "value", "dense",  # Attention layers
            "intermediate.dense", "output.dense"  # MLP layers
        ]
    elif architecture.lower() == 'swin':
        target_modules = [
            "query", "key", "value",  # Window attention
            "dense", "intermediate.dense", "output.dense"
        ]
    elif architecture.lower() == 'resnet':
        target_modules = [
            "conv1", "conv2", "conv3",  # Convolutional layers
            "downsample.0"  # Downsampling layers
        ]
    else:
        # Generic vision model
        target_modules = [
            "q_proj", "k_proj", "v_proj", "out_proj",
            "dense", "fc", "conv"
        ]
    
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        **kwargs
    )


def get_language_lora_config(
    architecture: str = "bert",
    rank: int = 16,
    alpha: Optional[int] = None,
    dropout: float = 0.1,
    **kwargs
) -> LoraConfig:
    """
    Get optimized LoRA configuration for language models.
    
    Args:
        architecture: Language architecture type ('bert', 'roberta', 'llama', 'gpt')
        rank: LoRA rank
        alpha: LoRA alpha (defaults to 2*rank)
        dropout: LoRA dropout rate
        **kwargs: Additional LoRA parameters
        
    Returns:
        LoraConfig optimized for language models
    """
    if alpha is None:
        alpha = rank * 2
    
    # Architecture-specific target modules
    if architecture.lower() in ['bert', 'roberta', 'deberta']:
        target_modules = [
            "query", "key", "value", "dense",  # Self-attention
            "intermediate.dense", "output.dense"  # Feed-forward
        ]
        task_type = TaskType.SEQ_CLS
        
    elif architecture.lower() in ['llama', 'mistral', 'qwen']:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",  # Self-attention
            "gate_proj", "up_proj", "down_proj"  # Feed-forward
        ]
        task_type = TaskType.CAUSAL_LM
        
    elif architecture.lower() in ['gpt', 'gpt2', 'gpt-neo']:
        target_modules = [
            "c_attn", "c_proj",  # Attention
            "c_fc"  # MLP
        ]
        task_type = TaskType.CAUSAL_LM
        
    elif architecture.lower() in ['t5', 'flan-t5', 'ul2']:
        target_modules = [
            "q", "k", "v", "o",  # Attention
            "wi_0", "wi_1", "wo"  # Feed-forward
        ]
        task_type = TaskType.SEQ_2_SEQ_LM
        
    else:
        # Generic language model
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "dense", "fc"
        ]
        task_type = TaskType.CAUSAL_LM
    
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type=task_type,
        **kwargs
    )


def get_multimodal_lora_config(
    model_type: str = "clip",
    vision_rank: int = 16,
    text_rank: int = 16,
    shared_rank: Optional[int] = None,
    alpha_ratio: float = 2.0,
    dropout: float = 0.1,
    **kwargs
) -> MultimodalLoRAConfig:
    """
    Get comprehensive LoRA configuration for multimodal models.
    
    Args:
        model_type: Multimodal model type ('clip', 'blip', 'llava', 'flamingo')
        vision_rank: LoRA rank for vision encoder
        text_rank: LoRA rank for text encoder
        shared_rank: LoRA rank for shared/fusion layers
        alpha_ratio: Ratio of alpha to rank
        dropout: LoRA dropout rate
        **kwargs: Additional parameters
        
    Returns:
        MultimodalLoRAConfig with separate configurations
    """
    # Vision encoder configuration
    if model_type.lower() == "clip":
        vision_config = get_vision_lora_config(
            "clip", vision_rank, int(vision_rank * alpha_ratio), dropout
        )\n        text_config = get_language_lora_config(\n            "bert", text_rank, int(text_rank * alpha_ratio), dropout\n        )\n        fusion_modules = ["visual_projection", "text_projection"]\n        \n    elif model_type.lower() == "blip":\n        vision_config = get_vision_lora_config(\n            "vit", vision_rank, int(vision_rank * alpha_ratio), dropout\n        )\n        text_config = get_language_lora_config(\n            "bert", text_rank, int(text_rank * alpha_ratio), dropout\n        )\n        fusion_modules = ["vision_proj", "text_proj", "itm_head", "itc_head"]\n        \n    elif model_type.lower() == "llava":\n        vision_config = get_vision_lora_config(\n            "clip", vision_rank, int(vision_rank * alpha_ratio), dropout\n        )\n        text_config = get_language_lora_config(\n            "llama", text_rank, int(text_rank * alpha_ratio), dropout\n        )\n        fusion_modules = ["mm_projector"]\n        \n    elif model_type.lower() == "flamingo":\n        vision_config = get_vision_lora_config(\n            "vit", vision_rank, int(vision_rank * alpha_ratio), dropout\n        )\n        text_config = get_language_lora_config(\n            "gpt", text_rank, int(text_rank * alpha_ratio), dropout\n        )\n        fusion_modules = ["perceiver", "gated_cross_attn"]\n        \n    else:\n        # Generic multimodal model\n        vision_config = get_vision_lora_config(\n            "vit", vision_rank, int(vision_rank * alpha_ratio), dropout\n        )\n        text_config = get_language_lora_config(\n            "bert", text_rank, int(text_rank * alpha_ratio), dropout\n        )\n        fusion_modules = ["fusion", "projector", "head"]\n    \n    # Shared/fusion layer configuration\n    shared_config = None\n    if shared_rank is not None and fusion_modules:\n        shared_config = LoraConfig(\n            r=shared_rank,\n            lora_alpha=int(shared_rank * alpha_ratio),\n            target_modules=fusion_modules,\n            lora_dropout=dropout,\n            bias="none",\n            task_type=TaskType.FEATURE_EXTRACTION\n        )\n    \n    return MultimodalLoRAConfig(\n        vision_config=vision_config,\n        text_config=text_config,\n        shared_config=shared_config,\n        fusion_modules=fusion_modules\n    )\n\n\ndef combine_lora_configs(\n    configs: List[LoraConfig],\n    merge_strategy: str = "union"\n) -> LoraConfig:\n    \"\"\"\n    Combine multiple LoRA configurations into a single config.\n    \n    Args:\n        configs: List of LoraConfig objects to combine\n        merge_strategy: How to merge configs ('union', 'intersection')\n        \n    Returns:\n        Combined LoraConfig\n    \"\"\"\n    if not configs:\n        raise ValueError(\"At least one config must be provided\")\n    \n    if len(configs) == 1:\n        return configs[0]\n    \n    # Use the first config as base\n    base_config = configs[0]\n    \n    # Combine target modules\n    all_modules = set(base_config.target_modules)\n    \n    for config in configs[1:]:\n        if merge_strategy == \"union\":\n            all_modules.update(config.target_modules)\n        elif merge_strategy == \"intersection\":\n            all_modules.intersection_update(config.target_modules)\n    \n    # Use average values for numerical parameters\n    avg_rank = int(sum(c.r for c in configs) / len(configs))\n    avg_alpha = int(sum(c.lora_alpha for c in configs) / len(configs))\n    avg_dropout = sum(c.lora_dropout for c in configs) / len(configs)\n    \n    return LoraConfig(\n        r=avg_rank,\n        lora_alpha=avg_alpha,\n        target_modules=list(all_modules),\n        lora_dropout=avg_dropout,\n        bias=base_config.bias,\n        task_type=base_config.task_type\n    )\n\n\ndef get_modality_specific_modules(\n    model: nn.Module,\n    modality: str = "vision"\n) -> List[str]:\n    \"\"\"\n    Extract modality-specific module names from a multimodal model.\n    \n    Args:\n        model: Multimodal model\n        modality: Target modality ('vision', 'text', 'fusion')\n        \n    Returns:\n        List of module names for the specified modality\n    \"\"\"\n    module_names = []\n    \n    # Modality-specific keywords\n    if modality == \"vision\":\n        keywords = [\n            \"vision\", \"visual\", \"image\", \"patch\", \"conv\",\n            \"vit\", \"swin\", \"resnet\", \"clip\"\n        ]\n    elif modality == \"text\":\n        keywords = [\n            \"text\", \"language\", \"bert\", \"roberta\", \"llama\",\n            \"token\", \"embed\", \"word\"\n        ]\n    elif modality == \"fusion\":\n        keywords = [\n            \"fusion\", \"cross\", \"multi\", \"projector\", \"head\",\n            \"itm\", \"itc\", \"mm\", \"multimodal\"\n        ]\n    else:\n        raise ValueError(f\"Unknown modality: {modality}\")\n    \n    # Find matching modules\n    for name, module in model.named_modules():\n        if isinstance(module, nn.Linear):\n            name_lower = name.lower()\n            if any(keyword in name_lower for keyword in keywords):\n                # Extract the final module name\n                module_name = name.split('.')[-1]\n                if module_name not in module_names:\n                    module_names.append(module_name)\n    \n    return module_names\n\n\ndef optimize_multimodal_lora(\n    model: nn.Module,\n    model_type: str,\n    performance_budget: float = 0.95,\n    memory_budget_mb: Optional[float] = None,\n    balance_modalities: bool = True\n) -> MultimodalLoRAConfig:\n    \"\"\"\n    Optimize LoRA configuration for multimodal models with resource constraints.\n    \n    Args:\n        model: Multimodal model to optimize for\n        model_type: Type of multimodal model\n        performance_budget: Target performance retention\n        memory_budget_mb: Memory budget in MB\n        balance_modalities: Whether to balance ranks across modalities\n        \n    Returns:\n        Optimized MultimodalLoRAConfig\n    \"\"\"\n    # Analyze model to determine optimal ranks\n    total_params = sum(p.numel() for p in model.parameters())\n    \n    # Base rank estimation\n    if total_params < 100e6:  # <100M\n        base_vision_rank = 8\n        base_text_rank = 8\n    elif total_params < 1e9:  # <1B\n        base_vision_rank = 16\n        base_text_rank = 16\n    elif total_params < 7e9:  # <7B\n        base_vision_rank = 32\n        base_text_rank = 32\n    else:  # >=7B\n        base_vision_rank = 64\n        base_text_rank = 64\n    \n    # Adjust based on performance budget\n    budget_factor = min(2.0, performance_budget / 0.8)\n    vision_rank = int(base_vision_rank * budget_factor)\n    text_rank = int(base_text_rank * budget_factor)\n    \n    # Balance modalities if requested\n    if balance_modalities:\n        # For vision-language models, text usually needs higher rank\n        if model_type.lower() in [\"clip\", \"blip\"]:\n            text_rank = int(text_rank * 1.2)\n        elif model_type.lower() == \"llava\":\n            # LLaVA is more text-heavy\n            text_rank = int(text_rank * 1.5)\n    \n    # Memory constraint adjustment\n    if memory_budget_mb:\n        # Rough estimate of LoRA memory usage\n        estimated_lora_params = (vision_rank + text_rank) * 100_000  # Rough estimate\n        estimated_memory_mb = estimated_lora_params * 4 / (1024**2)  # 4 bytes per param\n        \n        current_model_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)\n        available_memory = memory_budget_mb - current_model_mb\n        \n        if estimated_memory_mb > available_memory:\n            reduction_factor = available_memory / estimated_memory_mb\n            vision_rank = max(4, int(vision_rank * reduction_factor))\n            text_rank = max(4, int(text_rank * reduction_factor))\n    \n    # Shared layers rank (typically smaller)\n    shared_rank = max(4, min(vision_rank, text_rank) // 2)\n    \n    return get_multimodal_lora_config(\n        model_type=model_type,\n        vision_rank=vision_rank,\n        text_rank=text_rank,\n        shared_rank=shared_rank,\n        dropout=0.1 if total_params > 1e9 else 0.05\n    )\n\n\ndef print_multimodal_config_summary(config: MultimodalLoRAConfig) -> None:\n    \"\"\"\n    Print a summary of the multimodal LoRA configuration.\n    \n    Args:\n        config: MultimodalLoRAConfig to summarize\n    \"\"\"\n    print(\"\\n\" + \"=\" * 60)\n    print(\"MULTIMODAL LORA CONFIGURATION SUMMARY\")\n    print(\"=\" * 60)\n    \n    if config.vision_config:\n        print(f\"\\n🖼️  Vision Encoder Config:\")\n        print(f\"   Rank: {config.vision_config.r}\")\n        print(f\"   Alpha: {config.vision_config.lora_alpha}\")\n        print(f\"   Dropout: {config.vision_config.lora_dropout}\")\n        print(f\"   Target Modules: {config.vision_config.target_modules}\")\n    \n    if config.text_config:\n        print(f\"\\n📝 Text Encoder Config:\")\n        print(f\"   Rank: {config.text_config.r}\")\n        print(f\"   Alpha: {config.text_config.lora_alpha}\")\n        print(f\"   Dropout: {config.text_config.lora_dropout}\")\n        print(f\"   Target Modules: {config.text_config.target_modules}\")\n    \n    if config.shared_config:\n        print(f\"\\n🔗 Fusion/Shared Config:\")\n        print(f\"   Rank: {config.shared_config.r}\")\n        print(f\"   Alpha: {config.shared_config.lora_alpha}\")\n        print(f\"   Dropout: {config.shared_config.lora_dropout}\")\n        print(f\"   Target Modules: {config.shared_config.target_modules}\")\n    \n    if config.fusion_modules:\n        print(f\"\\n🎯 Fusion Modules: {config.fusion_modules}\")\n    \n    print(\"\\n\" + \"=\" * 60)\n\n\n# Example usage and testing\nif __name__ == \"__main__\":\n    print(\"Testing Multimodal LoRA Utilities\")\n    \n    # Test configuration generation\n    clip_config = get_multimodal_lora_config(\"clip\", vision_rank=16, text_rank=16)\n    print_multimodal_config_summary(clip_config)\n    \n    # Test individual configs\n    vision_config = get_vision_lora_config(\"clip\", rank=16)\n    text_config = get_language_lora_config(\"bert\", rank=16)\n    \n    # Test config combination\n    combined_config = combine_lora_configs([vision_config, text_config], \"union\")\n    print(f\"\\n🔄 Combined Config Modules: {combined_config.target_modules}\")"