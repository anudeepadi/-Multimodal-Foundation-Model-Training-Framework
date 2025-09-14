"""
Model quantization utilities supporting AWQ, GPTQ, and BitsAndBytes.
Enables efficient inference and training with reduced memory footprint.
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
import logging

logger = logging.getLogger(__name__)


class QuantizedModelLoader:
    """
    Unified interface for loading quantized models with different techniques.
    Supports AWQ, GPTQ, and BitsAndBytes quantization methods.
    """
    
    def __init__(self, quantization_method: str = "bnb"):
        """
        Initialize quantized model loader.
        
        Args:
            quantization_method: Quantization method ("bnb", "awq", "gptq")
        """
        self.quantization_method = quantization_method.lower()
        self.supported_methods = ["bnb", "awq", "gptq"]
        
        if self.quantization_method not in self.supported_methods:
            raise ValueError(f"Unsupported quantization method: {quantization_method}")
    
    def load_model(
        self,
        model_name: str,
        quantization_config: Optional[Dict[str, Any]] = None,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.float16,
        **kwargs
    ) -> tuple:
        """
        Load quantized model based on specified method.
        
        Args:
            model_name: HuggingFace model identifier
            quantization_config: Quantization-specific configuration
            device_map: Device mapping strategy
            torch_dtype: Model data type
            **kwargs: Additional model loading arguments
            
        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading model {model_name} with {self.quantization_method} quantization")
        
        if self.quantization_method == "bnb":
            return self._load_bnb_model(model_name, quantization_config, device_map, torch_dtype, **kwargs)
        elif self.quantization_method == "awq":
            return self._load_awq_model(model_name, quantization_config, device_map, torch_dtype, **kwargs)
        elif self.quantization_method == "gptq":
            return self._load_gptq_model(model_name, quantization_config, device_map, torch_dtype, **kwargs)
    
    def _load_bnb_model(
        self,
        model_name: str,
        quantization_config: Optional[Dict[str, Any]],
        device_map: str,
        torch_dtype: torch.dtype,
        **kwargs
    ) -> tuple:
        """Load model with BitsAndBytes quantization."""
        # Default 4-bit quantization config
        default_bnb_config = {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": torch_dtype,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4"
        }
        
        if quantization_config:
            default_bnb_config.update(quantization_config)
        
        bnb_config = BitsAndBytesConfig(**default_bnb_config)
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            **kwargs
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        return model, tokenizer
    
    def _load_awq_model(
        self,
        model_name: str,
        quantization_config: Optional[Dict[str, Any]],
        device_map: str,
        torch_dtype: torch.dtype,
        **kwargs
    ) -> tuple:
        """Load model with AWQ quantization."""
        try:
            from awq import AutoAWQForCausalLM
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError("AutoAWQ not installed. Install with: pip install autoawq")
        
        # Default AWQ config
        default_awq_config = {
            "fuse_layers": True,
            "batch_size": 1,
            "safetensors": True
        }
        
        if quantization_config:
            default_awq_config.update(quantization_config)
        
        model = AutoAWQForCausalLM.from_quantized(
            model_name,
            device_map=device_map,
            **default_awq_config,
            **kwargs
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        return model, tokenizer
    
    def _load_gptq_model(
        self,
        model_name: str,
        quantization_config: Optional[Dict[str, Any]],
        device_map: str,
        torch_dtype: torch.dtype,
        **kwargs
    ) -> tuple:
        """Load model with GPTQ quantization."""
        try:
            from optimum.gptq import GPTQQuantizer
        except ImportError:
            raise ImportError("Optimum GPTQ not installed. Install with: pip install optimum[gptq]")
        
        # Default GPTQ config
        default_gptq_config = {
            "bits": 4,
            "group_size": 128,
            "desc_act": False,
            "disable_exllama": False
        }
        
        if quantization_config:
            default_gptq_config.update(quantization_config)
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            **kwargs
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        return model, tokenizer
    
    def benchmark_quantization(
        self,
        model_name: str,
        test_prompts: list,
        methods: Optional[list] = None,
        max_new_tokens: int = 128
    ) -> Dict[str, Dict[str, Any]]:
        """
        Benchmark different quantization methods.
        
        Args:
            model_name: Model to benchmark
            test_prompts: List of test prompts
            methods: Quantization methods to compare
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Benchmark results dictionary
        """
        if methods is None:
            methods = ["bnb", "awq", "gptq"]
        
        results = {}
        
        for method in methods:
            logger.info(f"Benchmarking {method} quantization")
            
            try:
                # Load model with specific quantization
                self.quantization_method = method
                model, tokenizer = self.load_model(model_name)
                
                # Benchmark inference
                import time
                start_time = time.time()
                
                for prompt in test_prompts:
                    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=False
                        )
                
                end_time = time.time()
                
                # Calculate metrics
                model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3  # GB
                inference_time = end_time - start_time
                throughput = len(test_prompts) / inference_time
                
                results[method] = {
                    "model_size_gb": model_size,
                    "inference_time_s": inference_time,
                    "throughput_prompts_per_s": throughput,
                    "memory_usage_gb": torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                }
                
                # Clean up
                del model, tokenizer
                torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Error benchmarking {method}: {str(e)}")
                results[method] = {"error": str(e)}
        
        return results


class QuantizationUtils:
    """Utility functions for model quantization."""
    
    @staticmethod
    def estimate_model_size(model: nn.Module, dtype: torch.dtype = torch.float32) -> float:
        """Estimate model size in GB."""
        param_count = sum(p.numel() for p in model.parameters())
        bytes_per_param = torch.tensor([], dtype=dtype).element_size()
        size_gb = (param_count * bytes_per_param) / (1024 ** 3)
        return size_gb
    
    @staticmethod
    def compare_quantization_quality(
        original_model,
        quantized_model,
        tokenizer,
        test_dataset: list,
        metric_fn
    ) -> Dict[str, float]:
        """
        Compare quality between original and quantized models.
        
        Args:
            original_model: Original full-precision model
            quantized_model: Quantized model
            tokenizer: Model tokenizer
            test_dataset: Test prompts/data
            metric_fn: Function to compute quality metric
            
        Returns:
            Quality comparison metrics
        """
        results = {"original": [], "quantized": []}
        
        for data in test_dataset:
            # Generate with original model
            with torch.no_grad():
                orig_output = original_model.generate(
                    **tokenizer(data, return_tensors="pt"),
                    max_new_tokens=50,
                    do_sample=False
                )
                orig_text = tokenizer.decode(orig_output[0], skip_special_tokens=True)
                
                # Generate with quantized model
                quant_output = quantized_model.generate(
                    **tokenizer(data, return_tensors="pt"),
                    max_new_tokens=50,
                    do_sample=False
                )
                quant_text = tokenizer.decode(quant_output[0], skip_special_tokens=True)
            
            # Compute metrics
            results["original"].append(metric_fn(orig_text, data))
            results["quantized"].append(metric_fn(quant_text, data))
        
        # Average metrics
        avg_results = {
            "original_avg": sum(results["original"]) / len(results["original"]),
            "quantized_avg": sum(results["quantized"]) / len(results["quantized"]),
            "quality_retention": (sum(results["quantized"]) / sum(results["original"])) * 100
        }
        
        return avg_results
    
    @staticmethod
    def create_quantization_config(
        method: str,
        bits: int = 4,
        **kwargs
    ) -> Union[BitsAndBytesConfig, Dict[str, Any]]:
        """
        Create quantization configuration for specified method.
        
        Args:
            method: Quantization method ("bnb", "awq", "gptq")
            bits: Number of quantization bits
            **kwargs: Additional configuration parameters
            
        Returns:
            Quantization configuration object/dict
        """
        if method.lower() == "bnb":
            config = {
                "load_in_4bit": bits == 4,
                "load_in_8bit": bits == 8,
                "bnb_4bit_compute_dtype": torch.bfloat16,
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4"
            }
            config.update(kwargs)
            return BitsAndBytesConfig(**config)
        
        elif method.lower() == "awq":
            config = {
                "bits": bits,
                "fuse_layers": True,
                "batch_size": 1
            }
            config.update(kwargs)
            return config
        
        elif method.lower() == "gptq":
            config = {
                "bits": bits,
                "group_size": 128,
                "desc_act": False
            }
            config.update(kwargs)
            return config
        
        else:
            raise ValueError(f"Unsupported quantization method: {method}")


def create_quantized_model(config: Dict[str, Any]):
    """Factory function to create quantized model from configuration."""
    loader = QuantizedModelLoader(config.get("method", "bnb"))
    
    model, tokenizer = loader.load_model(
        model_name=config["model_name"],
        quantization_config=config.get("quantization_config"),
        device_map=config.get("device_map", "auto"),
        torch_dtype=getattr(torch, config.get("torch_dtype", "float16"))
    )
    
    return model, tokenizer, loader