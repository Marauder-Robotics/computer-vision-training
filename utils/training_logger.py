#!/usr/bin/env python3
"""
Self-managed training logger to replace WandB
Provides free, local/cloud storage of training metrics and graphs
"""
import json
import os
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from collections import defaultdict

class TrainingLogger:
    """Self-managed training logger that saves to DigitalOcean Spaces"""
    
    def __init__(self, 
                 project_name: str,
                 run_name: Optional[str] = None,
                 save_dir: str = "/datasets/marauder-do-bucket/training/logs",
                 config: Optional[Dict] = None):
        """
        Initialize training logger
        
        Args:
            project_name: Name of the project/experiment
            run_name: Specific run name (auto-generated if None)
            save_dir: Directory to save logs (DO bucket path)
            config: Configuration dictionary to log
        """
        self.project_name = project_name
        self.run_name = run_name or f"{project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Setup directories
        self.save_dir = Path(save_dir)
        self.run_dir = self.save_dir / self.project_name / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize log files
        self.metrics_file = self.run_dir / "metrics.jsonl"
        self.config_file = self.run_dir / "config.json"
        self.summary_file = self.run_dir / "summary.json"
        
        # Initialize data structures
        self.metrics_history = defaultdict(list)
        self.step = 0
        self.start_time = time.time()
        
        # Save config if provided
        if config:
            self.save_config(config)
    
    def save_config(self, config: Dict[str, Any]):
        """Save configuration to file"""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None, commit: bool = True):
        """
        Log metrics for current step
        
        Args:
            metrics: Dictionary of metric names and values
            step: Training step (auto-incremented if None)
            commit: Whether to write to disk immediately
        """
        if step is not None:
            self.step = step
        else:
            self.step += 1
        
        # Add timestamp
        metrics['_timestamp'] = datetime.now().isoformat()
        metrics['_step'] = self.step
        metrics['_runtime'] = time.time() - self.start_time
        
        # Store in history
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.metrics_history[key].append(value)
        
        # Write to file
        if commit:
            with open(self.metrics_file, 'a') as f:
                f.write(json.dumps(metrics, default=str) + '\n')
    
    def log_image(self, name: str, image: np.ndarray, step: Optional[int] = None):
        """Save an image to the run directory"""
        if step is None:
            step = self.step
        
        img_dir = self.run_dir / "images"
        img_dir.mkdir(exist_ok=True)
        
        # Save using matplotlib
        plt.figure(figsize=(10, 10))
        if len(image.shape) == 2:
            plt.imshow(image, cmap='gray')
        else:
            plt.imshow(image)
        plt.title(f"{name} - Step {step}")
        plt.axis('off')
        plt.savefig(img_dir / f"{name}_step_{step}.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_metrics(self, metric_names: Optional[List[str]] = None):
        """
        Generate and save plots for tracked metrics
        
        Args:
            metric_names: List of metrics to plot (all if None)
        """
        plots_dir = self.run_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        if metric_names is None:
            # Plot all numeric metrics except private ones
            metric_names = [k for k in self.metrics_history.keys() 
                          if not k.startswith('_') and len(self.metrics_history[k]) > 0]
        
        # Create individual plots
        for metric in metric_names:
            if metric in self.metrics_history:
                self._plot_single_metric(metric, plots_dir)
        
        # Create combined loss plot if applicable
        loss_metrics = [m for m in metric_names if 'loss' in m.lower()]
        if loss_metrics:
            self._plot_combined_metrics(loss_metrics, plots_dir / "combined_losses.png", "Losses")
        
        # Create combined accuracy plot if applicable
        acc_metrics = [m for m in metric_names if any(x in m.lower() for x in ['acc', 'map', 'precision', 'recall'])]
        if acc_metrics:
            self._plot_combined_metrics(acc_metrics, plots_dir / "combined_metrics.png", "Performance Metrics")
    
    def _plot_single_metric(self, metric_name: str, plots_dir: Path):
        """Plot a single metric"""
        values = self.metrics_history[metric_name]
        steps = self.metrics_history.get('_step', range(len(values)))[:len(values)]
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, values, 'b-', linewidth=2)
        plt.xlabel('Step')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} over Training')
        plt.grid(True, alpha=0.3)
        
        # Add smoothed line if enough data points
        if len(values) > 20:
            window = min(len(values) // 10, 50)
            smoothed = pd.Series(values).rolling(window, min_periods=1).mean()
            plt.plot(steps, smoothed, 'r-', alpha=0.5, linewidth=1, label='Smoothed')
            plt.legend()
        
        plt.savefig(plots_dir / f"{metric_name.replace('/', '_')}.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_combined_metrics(self, metric_names: List[str], output_path: Path, title: str):
        """Plot multiple metrics on the same graph"""
        plt.figure(figsize=(12, 8))
        
        for metric in metric_names:
            if metric in self.metrics_history:
                values = self.metrics_history[metric]
                steps = self.metrics_history.get('_step', range(len(values)))[:len(values)]
                plt.plot(steps, values, label=metric, linewidth=2)
        
        plt.xlabel('Step')
        plt.ylabel('Value')
        plt.title(title)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_summary(self, additional_info: Optional[Dict] = None):
        """Save training summary with final metrics"""
        summary = {
            'project_name': self.project_name,
            'run_name': self.run_name,
            'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
            'end_time': datetime.now().isoformat(),
            'total_runtime': time.time() - self.start_time,
            'total_steps': self.step,
            'final_metrics': {}
        }
        
        # Add final values for each metric
        for metric, values in self.metrics_history.items():
            if not metric.startswith('_') and values:
                summary['final_metrics'][metric] = {
                    'final': values[-1],
                    'best': max(values) if 'loss' not in metric.lower() else min(values),
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
        
        # Add additional info if provided
        if additional_info:
            summary.update(additional_info)
        
        # Save summary
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    
    def finish(self):
        """Finalize logging, generate plots and summary"""
        self.plot_metrics()
        self.save_summary()
        print(f"\n{'='*50}")
        print(f"Training completed!")
        print(f"Logs saved to: {self.run_dir}")
        print(f"Total runtime: {(time.time() - self.start_time)/3600:.2f} hours")
        print(f"{'='*50}\n")

class ExperimentTracker:
    """Track multiple experiments and compare results"""
    
    def __init__(self, base_dir: str = "/datasets/marauder-do-bucket/training/logs"):
        self.base_dir = Path(base_dir)
    
    def list_experiments(self, project_name: Optional[str] = None) -> List[Dict]:
        """List all experiments, optionally filtered by project"""
        experiments = []
        
        if project_name:
            project_dirs = [self.base_dir / project_name]
        else:
            project_dirs = [d for d in self.base_dir.iterdir() if d.is_dir()]
        
        for project_dir in project_dirs:
            if project_dir.exists():
                for run_dir in project_dir.iterdir():
                    if run_dir.is_dir():
                        summary_file = run_dir / "summary.json"
                        if summary_file.exists():
                            with open(summary_file) as f:
                                summary = json.load(f)
                                summary['path'] = str(run_dir)
                                experiments.append(summary)
        
        return experiments
    
    def compare_experiments(self, experiment_paths: List[str], metrics: List[str]):
        """Generate comparison plots for multiple experiments"""
        comparison_dir = self.base_dir / "comparisons" / datetime.now().strftime('%Y%m%d_%H%M%S')
        comparison_dir.mkdir(parents=True, exist_ok=True)
        
        # Load metrics for each experiment
        experiment_data = {}
        for path in experiment_paths:
            run_dir = Path(path)
            if (run_dir / "metrics.jsonl").exists():
                metrics_data = []
                with open(run_dir / "metrics.jsonl") as f:
                    for line in f:
                        metrics_data.append(json.loads(line))
                experiment_data[run_dir.name] = pd.DataFrame(metrics_data)
        
        # Create comparison plots
        for metric in metrics:
            plt.figure(figsize=(12, 8))
            
            for exp_name, df in experiment_data.items():
                if metric in df.columns:
                    plt.plot(df['_step'], df[metric], label=exp_name, linewidth=2)
            
            plt.xlabel('Step')
            plt.ylabel(metric)
            plt.title(f'{metric} Comparison')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(comparison_dir / f"{metric.replace('/', '_')}_comparison.png", 
                       dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"Comparison plots saved to: {comparison_dir}")
        return comparison_dir
