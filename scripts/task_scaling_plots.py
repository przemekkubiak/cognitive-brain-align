"""
Generate task-specific scaling plots with error bars for Pythia models.

"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Pythia model specifications (only these models to plot)
PYTHIA_MODELS = {
    "160m": 160e6,
    "410m": 410e6,
    "1b": 1e9,
    "1.4b": 1.4e9,
    "2.8b": 2.8e9,
    "6.9b": 6.9e9,
    "12b": 12e9,
}

# Task baselines for human-normalized scoring
# Format: {'random': value, 'human': value}
TASK_BASELINES = {
    'BART': {'random': 0.5, 'human': 0.95},  # Random: 50% burst, Human: expert doesn't burst
    'HorizonTask': {'random': 50.0, 'human': 75.0},  # Random: mid-range reward, Human: optimized choice
    'ProbabilisticReasoning': {'random': 0.5, 'human': 0.90},  # Random: 50% accuracy, Human: expert reasoning
    'RestlessBandit': {'random': 50.0, 'human': 75.0},  # Random: baseline reward, Human: optimized strategy
    'TwoStepTask': {'random': 0.5, 'human': 0.85},  # Random: 50% accuracy, Human: planning advantage
    'InstrumentalLearning': {'random': 0.5, 'human': 0.85},
    'TemporalDiscounting': {'random': 0.5, 'human': 0.85},
}

TASKS = [
    "BART",
    "HorizonTask", 
    "InstrumentalLearning",
    "ProbabilisticReasoning",
    "RestlessBandit",
    "TemporalDiscounting",
    "TwoStepTask",
]


def load_task_results(results_base: Path) -> Dict[str, pd.DataFrame]:
    """Load results for each task, aggregating all Pythia model runs."""
    task_data = {}
    
    for task in TASKS:
        task_dir = results_base / "Experiments" / task / "data" / "hf_EleutherAI"
        
        if not task_dir.exists():
            print(f"  Warning: Task directory not found: {task_dir}")
            continue
        
        model_results = []
        
        # Load CSV files for all Pythia models
        for model_name in PYTHIA_MODELS.keys():
            csv_files = list(task_dir.glob(f"pythia-{model_name}*.csv"))
            
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    df['model'] = model_name
                    df['model_params'] = PYTHIA_MODELS[model_name]
                    model_results.append(df)
                except Exception as e:
                    print(f"    Error loading {csv_file}: {e}")
        
        if model_results:
            task_data[task] = pd.concat(model_results, ignore_index=True)
            print(f"  {task}: {len(task_data[task])} total trials")
        else:
            print(f"  {task}: No data found")
    
    return task_data


def compute_task_metrics(df: pd.DataFrame, task_name: str) -> Dict[str, Dict]:
    """Compute performance metrics for each model on a task."""
    metrics = {}
    
    # Determine how to compute performance based on task and available columns
    if task_name == 'BART' and 'exploded' in df.columns:
        # BART: accuracy = 1 - exploded (didn't burst the balloon)
        df['accuracy'] = (~df['exploded']).astype(int)
        perf_col = 'accuracy'
    elif 'reward' in df.columns:
        # Tasks with reward column: HorizonTask, RestlessBandit
        perf_col = 'reward'
    elif 'left_pred' in df.columns and 'red_observation' in df.columns:
        # ProbabilisticReasoning: compute accuracy
        df['accuracy'] = (df['left_pred'] == df['red_observation']).astype(int)
        perf_col = 'accuracy'
    elif 'accurate' in df.columns:
        # TwoStepTask: accuracy column
        perf_col = 'accurate'
    else:
        print(f"  Warning: Cannot determine performance metric for {task_name}")
        return metrics
    
    for model in PYTHIA_MODELS.keys():
        model_df = df[df['model'] == model]
        
        if len(model_df) == 0:
            continue
        
        # Calculate performance rate per run to get error bars
        runs = model_df['run'].unique()
        run_rates = []
        
        for run in runs:
            run_df = model_df[model_df['run'] == run]
            perf_rate = run_df[perf_col].mean() if len(run_df) > 0 else 0
            run_rates.append(perf_rate)
        
        run_rates = np.array(run_rates)
        
        metrics[model] = {
            'mean': float(np.mean(run_rates)),
            'std': float(np.std(run_rates)),
            'sem': float(np.std(run_rates) / np.sqrt(len(run_rates))),
            'n_runs': len(run_rates)
        }
    
    return metrics


def generate_task_plots(task_data: Dict[str, pd.DataFrame], output_dir: Path):
    """Generate scaling plots for each task."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    
    for task, df in task_data.items():
        print(f"Plotting {task}...")
        
        metrics = compute_task_metrics(df, task)
        
        # Only use models that have metrics (data available)
        models = [m for m in PYTHIA_MODELS.keys() if m in metrics]
        
        if not models:
            print(f"  Warning: No models with data for {task}")
            continue
        
        model_params_log = [np.log10(PYTHIA_MODELS[m]) for m in models]
        
        # Get baselines for normalization
        baseline = TASK_BASELINES.get(task, {'random': 0.5, 'human': 1.0})
        random_score = baseline['random']
        human_score = baseline['human']
        
        # Normalize metrics to [0, 1] scale
        normalized_means = []
        normalized_stds = []
        for model in models:
            raw_mean = metrics[model]['mean']
            raw_std = metrics[model]['std']
            
            # Normalize: (value - random) / (human - random)
            norm_mean = (raw_mean - random_score) / (human_score - random_score)
            norm_std = raw_std / (human_score - random_score)  # Scale std by denominator
            
            normalized_means.append(norm_mean)
            normalized_stds.append(norm_std)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plot data with error bars (using normalized std)
        ax.errorbar(
            model_params_log,
            normalized_means,
            yerr=normalized_stds,
            fmt='o-',
            capsize=8,
            capthick=2,
            markersize=8,
            linewidth=2,
            label='Mean ± Std',
            color='#2E86AB',
            ecolor='#2E86AB',
            alpha=0.8
        )
        
        # Add reference lines at 0 (random) and 1 (human)
        ax.axhline(y=0.0, color='red', linestyle='--', linewidth=2, label='Random (0.0)', alpha=0.7)
        ax.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='Human (1.0)', alpha=0.7)
        
        # Formatting
        ax.set_xlabel('Model Size (log10 parameters)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Human-Normalized Performance', fontsize=12, fontweight='bold')
        ax.set_title(f'{task} Performance vs Model Size (0=Random, 1=Human)', fontsize=14, fontweight='bold')
        
        # Set y-axis limits with padding
        y_min = min(normalized_means) - 0.1 * max(normalized_stds + [0.1])
        y_max = max(normalized_means) + 0.1 * max(normalized_stds + [0.1])
        
        # Ensure we show the full 0-1 range for context
        y_min = min(y_min, -0.1)
        y_max = max(y_max, 1.2)
        
        ax.set_ylim(y_min, y_max)
        
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, loc='best')
        
        # Add model labels on x-axis
        ax.set_xticks(model_params_log)
        ax.set_xticklabels(models, rotation=45, ha='right')
        
        # Save plot
        plot_file = output_dir / f"task_{task}_scaling.png"
        plt.tight_layout()
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {plot_file}")
        print(f"  Y-axis range: {y_min:.2f} - {y_max:.2f} (human-normalized)")
        print(f"  Models in plot: {', '.join(models)} ({len(models)} models)")
        
        # Print metrics summary (both raw and normalized)
        print(f"  Metrics Summary (Raw → Normalized):")
        for model in models:
            raw_mean = metrics[model]['mean']
            raw_std = metrics[model]['std']
            norm_mean = (raw_mean - random_score) / (human_score - random_score)
            norm_std = raw_std / (human_score - random_score)
            print(f"    {model:6s}: {raw_mean:.4f}±{raw_std:.4f} → {norm_mean:.4f}±{norm_std:.4f}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate task-specific scaling plots')
    parser.add_argument('--results_dir', type=Path, default=Path('results'),
                        help='Path to results directory')
    parser.add_argument('--output_dir', type=Path, default=Path('results/task_analysis_output'),
                        help='Output directory for plots (default: results/task_analysis_output)')
    
    args = parser.parse_args()
    
    print("\nLoading task-specific results for Pythia models...")
    task_data = load_task_results(args.results_dir)
    
    print(f"\nLoaded {len(task_data)} tasks")
    for task in sorted(task_data.keys()):
        print(f"  - {task}")
    
    print(f"\nGenerating plots...")
    generate_task_plots(task_data, args.output_dir)
    
    print(f"\nPlots saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
