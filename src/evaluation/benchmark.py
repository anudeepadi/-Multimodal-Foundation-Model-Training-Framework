"""
Comprehensive benchmarking suite for multimodal models.
Includes performance, memory, throughput, and quality benchmarks.
"""
import torch
import torch.nn as nn
import numpy as np
import time
import psutil
import gc
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    model_name: str
    task: str
    metric_name: str
    value: float
    unit: str
    timestamp: str
    config: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelBenchmark:
    """
    Comprehensive model benchmarking suite.
    Evaluates performance across multiple dimensions.
    """
    
    def __init__(
        self,
        output_dir: str = "./benchmark_results",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize benchmark suite.
        
        Args:
            output_dir: Directory to save benchmark results
            device: Device to run benchmarks on
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.results = []
    
    def run_full_benchmark(
        self,
        models: Dict[str, nn.Module],
        test_data: Dict[str, Any],
        tasks: List[str] = None
    ) -> Dict[str, List[BenchmarkResult]]:
        """
        Run comprehensive benchmark suite.
        
        Args:
            models: Dictionary of models to benchmark
            test_data: Test data for evaluation
            tasks: List of tasks to benchmark
            
        Returns:
            Dictionary of benchmark results by task
        """
        if tasks is None:
            tasks = ["inference_speed", "memory_usage", "throughput", "quality"]
        
        all_results = {}
        
        for task in tasks:
            logger.info(f"Running {task} benchmark...")
            
            if task == "inference_speed":
                results = self._benchmark_inference_speed(models, test_data)
            elif task == "memory_usage":
                results = self._benchmark_memory_usage(models, test_data)
            elif task == "throughput":
                results = self._benchmark_throughput(models, test_data)
            elif task == "quality":
                results = self._benchmark_quality(models, test_data)
            else:
                logger.warning(f"Unknown benchmark task: {task}")
                continue
            
            all_results[task] = results
            self.results.extend(results)
        
        # Save results
        self._save_results()
        self._generate_report()
        
        return all_results
    
    def _benchmark_inference_speed(
        self,
        models: Dict[str, nn.Module],
        test_data: Dict[str, Any]
    ) -> List[BenchmarkResult]:
        """Benchmark inference speed."""
        results = []
        
        # Prepare test batch
        if 'batch' in test_data:
            batch = test_data['batch']
        else:
            # Create synthetic batch
            batch = {
                'input_ids': torch.randint(0, 1000, (8, 512)).to(self.device),
                'pixel_values': torch.randn(8, 3, 224, 224).to(self.device),
                'attention_mask': torch.ones(8, 512).to(self.device)
            }
        
        for model_name, model in models.items():
            model.eval()
            model.to(self.device)
            
            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    _ = model(**batch)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            
            # Benchmark
            times = []
            with torch.no_grad():
                for _ in range(100):
                    start_time = time.time()
                    _ = model(**batch)
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    end_time = time.time()
                    times.append(end_time - start_time)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            
            results.append(BenchmarkResult(
                model_name=model_name,
                task="inference_speed",
                metric_name="avg_inference_time",
                value=avg_time,
                unit="seconds",
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                config={"batch_size": batch['input_ids'].shape[0], "std": std_time}
            ))
            
            # Throughput (samples per second)
            throughput = batch['input_ids'].shape[0] / avg_time
            results.append(BenchmarkResult(
                model_name=model_name,
                task="inference_speed", 
                metric_name="throughput",
                value=throughput,
                unit="samples/second",
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                config={"batch_size": batch['input_ids'].shape[0]}
            ))
        
        return results
    
    def _benchmark_memory_usage(
        self,
        models: Dict[str, nn.Module],
        test_data: Dict[str, Any]
    ) -> List[BenchmarkResult]:
        """Benchmark memory usage."""
        results = []
        
        if 'batch' in test_data:
            batch = test_data['batch']
        else:
            batch = {
                'input_ids': torch.randint(0, 1000, (8, 512)).to(self.device),
                'pixel_values': torch.randn(8, 3, 224, 224).to(self.device),
                'attention_mask': torch.ones(8, 512).to(self.device)
            }
        
        for model_name, model in models.items():
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # Measure model size
            model_params = sum(p.numel() for p in model.parameters())
            model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
            
            results.append(BenchmarkResult(
                model_name=model_name,
                task="memory_usage",
                metric_name="model_size",
                value=model_size_mb,
                unit="MB",
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                config={"num_parameters": model_params}
            ))
            
            # Measure GPU memory usage during inference
            if torch.cuda.is_available():
                model.to(self.device)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                with torch.no_grad():
                    _ = model(**batch)
                
                peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
                
                results.append(BenchmarkResult(
                    model_name=model_name,
                    task="memory_usage",
                    metric_name="peak_gpu_memory",
                    value=peak_memory_mb,
                    unit="MB",
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                    config={"batch_size": batch['input_ids'].shape[0]}
                ))
            
            # Measure RAM usage
            process = psutil.Process()
            ram_usage_mb = process.memory_info().rss / 1024**2
            
            results.append(BenchmarkResult(
                model_name=model_name,
                task="memory_usage",
                metric_name="ram_usage",
                value=ram_usage_mb,
                unit="MB", 
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                config={}
            ))
        
        return results
    
    def _benchmark_throughput(
        self,
        models: Dict[str, nn.Module],
        test_data: Dict[str, Any]
    ) -> List[BenchmarkResult]:
        """Benchmark throughput across different batch sizes."""
        results = []
        batch_sizes = [1, 4, 8, 16, 32] if torch.cuda.is_available() else [1, 2, 4, 8]
        
        for model_name, model in models.items():
            model.eval()
            model.to(self.device)
            
            for batch_size in batch_sizes:
                try:
                    # Create batch
                    batch = {
                        'input_ids': torch.randint(0, 1000, (batch_size, 512)).to(self.device),
                        'pixel_values': torch.randn(batch_size, 3, 224, 224).to(self.device),
                        'attention_mask': torch.ones(batch_size, 512).to(self.device)
                    }
                    
                    # Warmup
                    with torch.no_grad():
                        for _ in range(5):
                            _ = model(**batch)
                    
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    
                    # Benchmark
                    times = []
                    with torch.no_grad():
                        for _ in range(50):
                            start_time = time.time()
                            _ = model(**batch)
                            torch.cuda.synchronize() if torch.cuda.is_available() else None
                            end_time = time.time()
                            times.append(end_time - start_time)
                    
                    avg_time = np.mean(times)
                    throughput = batch_size / avg_time
                    
                    results.append(BenchmarkResult(
                        model_name=model_name,
                        task="throughput",
                        metric_name=f"throughput_bs_{batch_size}",
                        value=throughput,
                        unit="samples/second",
                        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                        config={"batch_size": batch_size, "avg_time": avg_time}
                    ))
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.warning(f"OOM for {model_name} with batch size {batch_size}")
                        break
                    else:
                        raise e
        
        return results
    
    def _benchmark_quality(
        self,
        models: Dict[str, nn.Module],
        test_data: Dict[str, Any]
    ) -> List[BenchmarkResult]:
        """Benchmark model quality metrics."""
        results = []
        
        if 'quality_data' not in test_data:
            logger.warning("No quality data provided, skipping quality benchmark")
            return results
        
        quality_data = test_data['quality_data']
        
        for model_name, model in models.items():
            model.eval()
            
            # Run quality evaluation
            if hasattr(model, 'generate') and 'images' in quality_data and 'captions' in quality_data:
                # Caption generation quality
                predictions = []
                
                with torch.no_grad():
                    for image in quality_data['images']:
                        # Generate caption
                        if hasattr(model, 'processor'):
                            inputs = model.processor(images=[image], return_tensors="pt").to(self.device)
                        else:
                            # Assume model has a specific format
                            inputs = {'pixel_values': torch.randn(1, 3, 224, 224).to(self.device)}
                        
                        outputs = model.generate(**inputs, max_new_tokens=50)
                        if hasattr(model, 'processor'):
                            caption = model.processor.decode(outputs[0], skip_special_tokens=True)
                        else:
                            caption = "Generated caption"  # Placeholder
                        
                        predictions.append(caption)
                
                # Compute metrics (simplified example)
                from .metrics import compute_captioning_metrics
                
                references = [[ref] for ref in quality_data['captions']]
                metrics = compute_captioning_metrics(predictions, references, quality_data['images'])
                
                for metric_name, value in metrics.items():
                    results.append(BenchmarkResult(
                        model_name=model_name,
                        task="quality",
                        metric_name=metric_name,
                        value=value,
                        unit="score",
                        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                        config={"num_samples": len(predictions)}
                    ))
        
        return results
    
    def _save_results(self):
        """Save benchmark results."""
        results_data = [result.to_dict() for result in self.results]
        
        # Save as JSON
        with open(self.output_dir / "benchmark_results.json", 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save as CSV
        df = pd.DataFrame(results_data)
        df.to_csv(self.output_dir / "benchmark_results.csv", index=False)
        
        logger.info(f"Benchmark results saved to {self.output_dir}")
    
    def _generate_report(self):
        """Generate benchmark report with visualizations."""
        df = pd.DataFrame([result.to_dict() for result in self.results])
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Inference speed comparison
        speed_data = df[df['task'] == 'inference_speed'][df['metric_name'] == 'avg_inference_time']
        if not speed_data.empty:
            sns.barplot(data=speed_data, x='model_name', y='value', ax=axes[0, 0])
            axes[0, 0].set_title('Average Inference Time')
            axes[0, 0].set_ylabel('Time (seconds)')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Memory usage comparison
        memory_data = df[df['task'] == 'memory_usage'][df['metric_name'] == 'model_size']
        if not memory_data.empty:
            sns.barplot(data=memory_data, x='model_name', y='value', ax=axes[0, 1])
            axes[0, 1].set_title('Model Size')
            axes[0, 1].set_ylabel('Size (MB)')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Throughput comparison
        throughput_data = df[df['task'] == 'inference_speed'][df['metric_name'] == 'throughput']
        if not throughput_data.empty:
            sns.barplot(data=throughput_data, x='model_name', y='value', ax=axes[1, 0])
            axes[1, 0].set_title('Throughput')
            axes[1, 0].set_ylabel('Samples/second')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Quality metrics (if available)
        quality_data = df[df['task'] == 'quality']
        if not quality_data.empty:
            # Plot first quality metric
            first_metric = quality_data['metric_name'].iloc[0]
            metric_data = quality_data[quality_data['metric_name'] == first_metric]
            sns.barplot(data=metric_data, x='model_name', y='value', ax=axes[1, 1])
            axes[1, 1].set_title(f'Quality: {first_metric}')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].tick_params(axis='x', rotation=45)
        else:
            axes[1, 1].text(0.5, 0.5, 'No quality data', ha='center', va='center')
            axes[1, 1].set_title('Quality Metrics')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "benchmark_report.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate summary report
        self._generate_summary_report(df)
    
    def _generate_summary_report(self, df: pd.DataFrame):
        """Generate text summary report."""
        report = []
        report.append("# Multimodal Model Benchmark Report\n")
        report.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Model overview
        models = df['model_name'].unique()
        report.append(f"## Models Benchmarked\n")
        for model in models:
            report.append(f"- {model}")
        report.append("\n")
        
        # Performance summary
        report.append("## Performance Summary\n")
        
        # Inference speed
        speed_data = df[(df['task'] == 'inference_speed') & (df['metric_name'] == 'avg_inference_time')]
        if not speed_data.empty:
            report.append("### Inference Speed\n")
            for _, row in speed_data.iterrows():
                report.append(f"- {row['model_name']}: {row['value']:.4f} seconds per batch\n")
            report.append("\n")
        
        # Memory usage
        memory_data = df[(df['task'] == 'memory_usage') & (df['metric_name'] == 'model_size')]
        if not memory_data.empty:
            report.append("### Model Size\n")
            for _, row in memory_data.iterrows():
                report.append(f"- {row['model_name']}: {row['value']:.1f} MB\n")
            report.append("\n")
        
        # Throughput
        throughput_data = df[(df['task'] == 'inference_speed') & (df['metric_name'] == 'throughput')]
        if not throughput_data.empty:
            report.append("### Throughput\n")
            for _, row in throughput_data.iterrows():
                report.append(f"- {row['model_name']}: {row['value']:.2f} samples/second\n")
            report.append("\n")
        
        # Quality metrics
        quality_data = df[df['task'] == 'quality']
        if not quality_data.empty:
            report.append("### Quality Metrics\n")
            for metric in quality_data['metric_name'].unique():
                report.append(f"#### {metric}\n")
                metric_data = quality_data[quality_data['metric_name'] == metric]
                for _, row in metric_data.iterrows():
                    report.append(f"- {row['model_name']}: {row['value']:.4f}\n")
                report.append("\n")
        
        # Recommendations
        report.append("## Recommendations\n")
        
        if not speed_data.empty:
            fastest_model = speed_data.loc[speed_data['value'].idxmin(), 'model_name']
            report.append(f"- **Fastest Model**: {fastest_model}\n")
        
        if not memory_data.empty:
            smallest_model = memory_data.loc[memory_data['value'].idxmin(), 'model_name']
            report.append(f"- **Most Memory Efficient**: {smallest_model}\n")
        
        if not throughput_data.empty:
            highest_throughput = throughput_data.loc[throughput_data['value'].idxmax(), 'model_name']
            report.append(f"- **Highest Throughput**: {highest_throughput}\n")
        
        # Save report
        with open(self.output_dir / "benchmark_report.md", 'w') as f:
            f.writelines(report)
        
        logger.info(f"Benchmark report generated: {self.output_dir}/benchmark_report.md")


class PerformanceBenchmark:
    """Focused performance benchmarking utilities."""
    
    @staticmethod
    def measure_inference_time(
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> Dict[str, float]:
        """Measure inference time with statistical analysis."""
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(**batch)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        # Measure
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.perf_counter()
                _ = model(**batch)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.perf_counter()
                times.append(end_time - start_time)
        
        times = np.array(times)
        
        return {
            'mean': float(np.mean(times)),
            'std': float(np.std(times)),
            'min': float(np.min(times)),
            'max': float(np.max(times)),
            'median': float(np.median(times)),
            'p95': float(np.percentile(times, 95)),
            'p99': float(np.percentile(times, 99))
        }


class MemoryBenchmark:
    """Memory usage benchmarking utilities."""
    
    @staticmethod
    def measure_model_memory(model: nn.Module) -> Dict[str, float]:
        """Measure model memory footprint."""
        model_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        
        return {
            'model_size_mb': model_size / 1024**2,
            'buffer_size_mb': buffer_size / 1024**2,
            'total_size_mb': (model_size + buffer_size) / 1024**2,
            'num_parameters': sum(p.numel() for p in model.parameters())
        }
    
    @staticmethod
    def measure_peak_memory(
        model: nn.Module,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Measure peak GPU memory usage during forward pass."""
        if not torch.cuda.is_available():
            return {'peak_memory_mb': 0, 'allocated_memory_mb': 0}
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            _ = model(**batch)
        
        peak_memory = torch.cuda.max_memory_allocated()
        allocated_memory = torch.cuda.memory_allocated()
        
        return {
            'peak_memory_mb': peak_memory / 1024**2,
            'allocated_memory_mb': allocated_memory / 1024**2
        }


class ThroughputBenchmark:
    """Throughput benchmarking utilities."""
    
    @staticmethod
    def measure_throughput(
        model: nn.Module,
        create_batch_fn: Callable[[int], Dict[str, torch.Tensor]],
        batch_sizes: List[int] = None,
        duration: float = 30.0
    ) -> Dict[int, Dict[str, float]]:
        """Measure throughput across different batch sizes."""
        if batch_sizes is None:
            batch_sizes = [1, 4, 8, 16, 32]
        
        model.eval()
        results = {}
        
        for batch_size in batch_sizes:
            try:
                batch = create_batch_fn(batch_size)
                
                # Warmup
                with torch.no_grad():
                    for _ in range(10):
                        _ = model(**batch)
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                
                # Measure for specified duration
                start_time = time.time()
                num_batches = 0
                
                with torch.no_grad():
                    while time.time() - start_time < duration:
                        _ = model(**batch)
                        torch.cuda.synchronize() if torch.cuda.is_available() else None
                        num_batches += 1
                
                elapsed_time = time.time() - start_time
                samples_per_second = (num_batches * batch_size) / elapsed_time
                batches_per_second = num_batches / elapsed_time
                
                results[batch_size] = {
                    'samples_per_second': samples_per_second,
                    'batches_per_second': batches_per_second,
                    'elapsed_time': elapsed_time,
                    'num_batches': num_batches
                }
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning(f"OOM at batch size {batch_size}")
                    break
                else:
                    raise e
        
        return results