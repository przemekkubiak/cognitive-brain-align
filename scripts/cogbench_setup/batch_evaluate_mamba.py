#!/usr/bin/env python3
"""
Batch evaluation script for all Mamba models on CogBench.

"""

import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime


MAMBA_MODELS = [
    ("130m", "state-spaces/mamba-130m-hf"),
    ("370m", "state-spaces/mamba-370m-hf"),
    ("790m", "state-spaces/mamba-790m-hf"),
    ("1.4b", "state-spaces/mamba-1.4b-hf"),
    ("2.8b", "state-spaces/mamba-2.8b-hf"),
]

# Remote machine configuration
REMOTE_HOST = "root@216.153.49.92"
COGBENCH_PATH = "/root/cognitive-brain-align/CogBench"


def run_model_evaluation(model_name: str, model_id: str) -> bool:
    """Run evaluation for a single model."""
    print(f"\n{'='*70}")
    print(f"EVALUATING: {model_name}")
    print(f"Model ID: {model_id}")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    engine_name = f"hf_{model_id}"
    cmd = [
        "ssh",
        REMOTE_HOST,
        f"cd {COGBENCH_PATH} && python3 full_run.py --engine {engine_name}"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, timeout=3600)
        print(f"\n{model_name} evaluation completed successfully")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"\n{model_name} evaluation failed")
        print(f"  Return code: {e.returncode}")
        return False
    
    except subprocess.TimeoutExpired:
        print(f"\n{model_name} evaluation timed out (>1 hour)")
        return False
    
    except Exception as e:
        print(f"\n Error evaluating {model_name}: {e}")
        return False


def sync_results_from_remote(model_name: str, local_results_dir: Path):
    """Sync results from remote machine after evaluation."""
    print(f"  Syncing results from remote...")
    
    remote_results = f"{REMOTE_HOST}:cognitive-brain-align/results/Experiments"
    local_path = local_results_dir / "Experiments"
    
    try:
        cmd = [
            "scp",
            "-r",
            f"{remote_results}",
            str(local_path),
        ]
        
        result = subprocess.run(cmd, capture_output=True, timeout=600, text=True)
        
        if result.returncode == 0:
            print(f"  [OK] Results synced successfully")
        else:
            print(f"  [WARNING] Warning: Sync had issues: {result.stderr}")
    
    except subprocess.TimeoutExpired:
        print(f"  [WARNING] Warning: Sync timed out")
    except Exception as e:
        print(f"  [WARNING] Warning: Sync failed: {e}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Batch evaluate all Mamba models on CogBench",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all models
  python batch_evaluate_mamba.py

  # Run specific models and sync results
  python batch_evaluate_mamba.py --models 130m 370m

  # Dry run (show what would run)
  python batch_evaluate_mamba.py --dry-run
""")
    
    parser.add_argument(
        "--models",
        nargs="+",
        default=[m[0] for m in MAMBA_MODELS],
        help="Models to evaluate (default: all models)"
    )
    
    parser.add_argument(
        "--local_results",
        type=Path,
        default=Path("results"),
        help="Local results directory for syncing (default: ./results)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing"
    )
    
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Don't sync results from remote after each model"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("MAMBA BATCH EVALUATION")
    print("=" * 70)
    print(f"Remote host: {REMOTE_HOST}")
    print(f"CogBench path: {COGBENCH_PATH}")
    print(f"Local results: {args.local_results}")
    
    # Filter models
    models_to_eval = []
    for model_name, model_id in MAMBA_MODELS:
        if model_name in args.models:
            models_to_eval.append((model_name, model_id))
    
    print(f"\nModels to evaluate ({len(models_to_eval)}):")
    for model_name, model_id in models_to_eval:
        print(f"  - {model_name:8s} ({model_id})")
    
    if args.dry_run:
        print("\n[DRY RUN] Commands that would be executed:")
        for model_name, model_id in models_to_eval:
            engine_name = f"hf_{model_id}"
            cmd = f"cd {COGBENCH_PATH} && python3 full_run.py --engine {engine_name}"
            print(f"  ssh {REMOTE_HOST} '{cmd}'")
        return
    
    # Run evaluations
    print(f"\n{'='*70}")
    print("STARTING BATCH EVALUATION")
    print(f"{'='*70}\n")
    
    results = {}
    start_time = datetime.now()
    
    for i, (model_name, model_id) in enumerate(models_to_eval):
        print(f"\n[{i+1}/{len(models_to_eval)}] Evaluating {model_name}")
        
        success = run_model_evaluation(model_name, model_id)
        results[model_name] = "completed" if success else "failed"
        
        # Sync results after each model
        if success and not args.no_sync:
            sync_results_from_remote(model_name, args.local_results)
        
        # Small delay between models
        if i < len(models_to_eval) - 1:
            print(f"\nWaiting 30 seconds before next model...")
            time.sleep(30)
    
    # Final summary
    elapsed = datetime.now() - start_time
    elapsed_hours = elapsed.total_seconds() / 3600
    
    print(f"\n{'='*70}")
    print("BATCH EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {elapsed_hours:.1f} hours")
    print(f"\nResults:")
    for model_name, status in results.items():
        print(f"  {model_name:8s}: {status}")
    
    # Count completions
    completed = sum(1 for s in results.values() if "completed" in s)
    failed = len(results) - completed
    
    print(f"\nSummary: {completed} completed, {failed} failed")
    print(f"Results synced to: {args.local_results}")


if __name__ == '__main__':
    main()
