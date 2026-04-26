"""
Analysis script for Pythia CogBench results.

Loads CSV results from all model evaluations and generates analysis.

Usage:
    python analyze_pythia_results.py --results_dir ../results/
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# Mapping of model names to parameter counts
PYTHIA_PARAMS = {
    "70m": 70e6,
    "160m": 160e6,
    "410m": 410e6,
    "1b": 1e9,
    "1.4b": 1.4e9,
    "2.8b": 2.8e9,
    "6.9b": 6.9e9,
    "12b": 12e9,
}


def extract_model_name(dirname: str) -> Optional[str]:
    """Extract model name from directory (e.g., 'pythia_70m' -> '70m')."""
    match = re.match(r"pythia_(\d+\.?\d*[bm]?)", dirname)
    return match.group(1) if match else None


def load_model_csv(csv_file: Path) -> Optional[pd.DataFrame]:
    """Load CSV results from a single model evaluation."""
    try:
        df = pd.read_csv(csv_file)
        return df
    except Exception as e:
        print(f"    Error loading {csv_file}: {e}")
        return None


def analyze_model_data(model_name: str, df: pd.DataFrame) -> Dict:
    """Extract key metrics from model's behavioral data."""
    analysis = {
        "model": model_name,
        "total_trials": len(df),
        "runs": df["run"].nunique() if "run" in df.columns else len(df),
        "columns": list(df.columns),
    }
    
    # Basic statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ["run", "trial"]:
            try:
                analysis[f"{col}_mean"] = float(df[col].mean())
                analysis[f"{col}_std"] = float(df[col].std())
            except:
                pass
    
    # Count reward outcomes if available
    if "reward" in df.columns:
        analysis["reward_rate"] = float((df["reward"] == 1).sum() / len(df))
    
    return analysis


def aggregate_results(results_dir: Path) -> Dict[str, Dict]:
    """
    Aggregate results from all model CSV files.
    
    Returns:
        Dictionary mapping model names to their analysis
    """
    aggregated = {}
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        print(f"  Looking for: {results_dir.absolute()}")
        return aggregated
    
    # Find all model subdirectories
    model_dirs = [d for d in results_dir.iterdir() 
                  if d.is_dir() and d.name.startswith("pythia_")]
    
    print(f"Found {len(model_dirs)} model evaluation directories\n")
    
    for model_dir in sorted(model_dirs):
        model_name = extract_model_name(model_dir.name)
        if not model_name:
            continue
        
        csv_file = model_dir / f"pythia-{model_name}.csv"
        if not csv_file.exists():
            print(f"  {model_name}: CSV not found ({csv_file.name})")
            continue
        
        print(f"  {model_name}...", end=" ")
        df = load_model_csv(csv_file)
        
        if df is not None:
            analysis = analyze_model_data(model_name, df)
            aggregated[model_name] = analysis
            print(f"[DONE] ({len(df)} trials, {analysis['runs']} runs)")
        else:
            print("[FAILED]")
    
    return aggregated


def compute_scaling_metrics(aggregated_results: Dict[str, Dict]) -> Tuple[Dict, List[Dict]]:
    """
    Compute scaling relationships between model size and performance.
    """
    scaling_analysis = {
        "models_evaluated": len(aggregated_results),
        "model_sizes_log10": [],
        "model_names": [],
        "metrics": {},
    }
    
    # Collect data points
    model_data = []
    for model_name in sorted(aggregated_results.keys(), 
                             key=lambda x: PYTHIA_PARAMS.get(x, 0)):
        params = PYTHIA_PARAMS.get(model_name, 0)
        results = aggregated_results[model_name]
        
        if params > 0:
            model_data.append({
                "name": model_name,
                "params": params,
                "log_params": np.log10(params),
                "total_trials": results.get("total_trials", 0),
                "reward_rate": results.get("reward_rate", None),
            })
    
    if model_data:
        scaling_analysis["model_names"] = [m["name"] for m in model_data]
        scaling_analysis["model_sizes_log10"] = [m["log_params"] for m in model_data]
        
        # Analyze reward rates if available
        reward_rates = [m["reward_rate"] for m in model_data if m["reward_rate"] is not None]
        if reward_rates:
            scaling_analysis["reward_rate_mean"] = float(np.mean(reward_rates))
            scaling_analysis["reward_rate_std"] = float(np.std(reward_rates))
            scaling_analysis["reward_rates"] = reward_rates
    
    return scaling_analysis, model_data


def print_analysis_summary(aggregated_results: Dict[str, Dict],
                          scaling_metrics: Dict,
                          model_data: List[Dict]):
    """Print summary of aggregated results."""
    print("\n" + "=" * 70)
    print("PYTHIA COGBENCH ANALYSIS SUMMARY")
    print("=" * 70)
    
    print(f"\nModels loaded: {len(aggregated_results)}")
    
    if aggregated_results:
        print("\nModel Statistics:")
        print("-" * 70)
        print(f"{'Model':<12} {'Params':<12} {'Trials':<10} {'Runs':<8} {'Reward Rate':<12}")
        print("-" * 70)
        
        for model_data_item in model_data:
            name = model_data_item["name"]
            params = model_data_item["params"]
            result = aggregated_results[name]
            
            trials = result.get("total_trials", "?")
            runs = result.get("runs", "?")
            reward_rate = result.get("reward_rate", None)
            
            reward_str = f"{reward_rate:.1%}" if reward_rate else "N/A"
            param_str = f"{params/1e9:.1f}B" if params >= 1e9 else f"{params/1e6:.0f}M"
            
            print(f"{name:<12} {param_str:<12} {trials:<10} {runs:<8} {reward_str:<12}")
        
        print("-" * 70)
    
    print(f"\nTotal models evaluated: {scaling_metrics['models_evaluated']}")
    
    if "reward_rate_mean" in scaling_metrics:
        print(f"Average reward rate: {scaling_metrics['reward_rate_mean']:.1%}")


def generate_plots(aggregated_results: Dict[str, Dict],
                  scaling_metrics: Dict,
                  model_data: List[Dict],
                  output_dir: Optional[Path] = None):
    """Generate visualization plots if matplotlib available."""
    
    if not HAS_MATPLOTLIB:
        print("\n(Plotting skipped - matplotlib not available)")
        return
    
    print("\nGenerating plots...")
    
    if not output_dir:
        output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)
    
    # Plot 1: Model size vs reward rate
    if model_data and "reward_rates" in scaling_metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        model_names = [m["name"] for m in model_data]
        model_sizes = [m["params"] / 1e9 for m in model_data]  # in billions
        reward_rates = scaling_metrics["reward_rates"]
        
        ax.plot(model_sizes, reward_rates, "o-", linewidth=2, markersize=8)
        ax.set_xlabel("Model Size (Billion Parameters)", fontsize=12)
        ax.set_ylabel("Reward Rate", fontsize=12)
        ax.set_title("Pythia Model Scaling: Reward Rate vs Model Size", fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Add model labels
        for name, size, rate in zip(model_names, model_sizes, reward_rates):
            ax.annotate(name, (size, rate), xytext=(5, 5), 
                       textcoords="offset points", fontsize=9)
        
        plot_file = output_dir / "scaling_reward_rate.png"
        fig.savefig(plot_file, dpi=150, bbox_inches="tight")
        print(f"  Saved: {plot_file}")
        plt.close(fig)
    
    print("[DONE] Plots generated")


def main():
    parser = argparse.ArgumentParser(description="Analyze Pythia CogBench results")
    parser.add_argument("--results_dir", type=Path, default=Path("../results"),
                       help="Directory containing model results")
    parser.add_argument("--output_dir", type=Path, default=Path("../results/analysis_output"),
                       help="Directory for analysis output (default: results/analysis_output)")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("PYTHIA COGBENCH RESULTS ANALYSIS")
    print("=" * 70)
    print(f"\nResults directory: {args.results_dir.absolute()}")
    
    # Aggregate results
    aggregated = aggregate_results(args.results_dir)
    
    if not aggregated:
        print("\nError: No results found to analyze")
        return 1
    
    # Compute scaling metrics
    scaling_metrics, model_data = compute_scaling_metrics(aggregated)
    
    # Print summary
    print_analysis_summary(aggregated, scaling_metrics, model_data)
    
    # Generate plots if requested
    if args.plot:
        generate_plots(aggregated, scaling_metrics, model_data, args.output_dir)
    
    # Save analysis results
    args.output_dir.mkdir(exist_ok=True)
    
    results_file = args.output_dir / "analysis_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "aggregated": {k: v for k, v in aggregated.items()},
            "scaling_metrics": scaling_metrics,
        }, f, indent=2, default=str)
    
    print(f"\n[DONE] Analysis saved to {results_file}")
    print("\n" + "=" * 70)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
