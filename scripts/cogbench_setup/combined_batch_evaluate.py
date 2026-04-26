#!/usr/bin/env python3
"""
Combined batch evaluation script for both Pythia and Mamba models on CogBench.

This script coordinates evaluation of multiple model families sequentially.
"""

import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime


# Model families
PYTHIA_MODELS = [
    ("160m", "EleutherAI/pythia-160m"),
    ("410m", "EleutherAI/pythia-410m"),
    ("1b", "EleutherAI/pythia-1b"),
    ("1.4b", "EleutherAI/pythia-1.4b"),
    ("2.8b", "EleutherAI/pythia-2.8b"),
    ("6.9b", "EleutherAI/pythia-6.9b"),
    ("12b", "EleutherAI/pythia-12b"),
]

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


def run_model_evaluation(family: str, model_name: str, model_id: str) -> bool:
    """Run evaluation for a single model."""
    print(f"\n{'='*70}")
    print(f"EVALUATING: {family.upper()} - {model_name}")
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
        result = subprocess.run(cmd, check=True, timeout=7200)  # 2 hour timeout
        print(f"\n[DONE] {family.upper()} {model_name} evaluation completed successfully")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"\n[FAILED] {family.upper()} {model_name} evaluation failed")
        print(f"  Return code: {e.returncode}")
        return False
    
    except subprocess.TimeoutExpired:
        print(f"\n[FAILED] {family.upper()} {model_name} evaluation timed out (>2 hours)")
        return False
    
    except Exception as e:
        print(f"\n[FAILED] Error evaluating {family.upper()} {model_name}: {e}")
        return False


def sync_results_from_remote(local_results_dir: Path):
    """Sync all results from remote machine."""
    print(f"\nSyncing all results from remote...")
    
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
            print(f"[OK] Results synced successfully to {local_path}")
        else:
            print(f"[WARNING] Warning: Sync had issues: {result.stderr}")
    
    except subprocess.TimeoutExpired:
        print(f"[WARNING] Warning: Sync timed out")
    except Exception as e:
        print(f"[WARNING] Warning: Sync failed: {e}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Batch evaluate Pythia and Mamba models on CogBench",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all Pythia models
  python combined_batch_evaluate.py --families pythia

  # Evaluate all Mamba models
  python combined_batch_evaluate.py --families mamba

  # Evaluate both families
  python combined_batch_evaluate.py --families pythia mamba

  # Evaluate specific models
  python combined_batch_evaluate.py --pythia 160m 410m --mamba 130m 370m

  # Dry run
  python combined_batch_evaluate.py --families pythia mamba --dry-run
""")
    
    parser.add_argument(
        "--families",
        nargs="+",
        choices=["pythia", "mamba"],
        default=["pythia", "mamba"],
        help="Model families to evaluate (default: both)"
    )
    
    parser.add_argument(
        "--pythia",
        nargs="+",
        default=[m[0] for m in PYTHIA_MODELS],
        help="Pythia models to evaluate"
    )
    
    parser.add_argument(
        "--mamba",
        nargs="+",
        default=[m[0] for m in MAMBA_MODELS],
        help="Mamba models to evaluate"
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
        help="Don't sync results from remote"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("COMBINED MODEL FAMILY BATCH EVALUATION")
    print("=" * 70)
    print(f"Remote host: {REMOTE_HOST}")
    print(f"CogBench path: {COGBENCH_PATH}")
    print(f"Local results: {args.local_results}")
    
    # Build model list
    all_models = []
    
    if "pythia" in args.families:
        pythia_models = [
            (m[0], m[1]) for m in PYTHIA_MODELS if m[0] in args.pythia
        ]
        all_models.extend([("pythia", m[0], m[1]) for m in pythia_models])
    
    if "mamba" in args.families:
        mamba_models = [
            (m[0], m[1]) for m in MAMBA_MODELS if m[0] in args.mamba
        ]
        all_models.extend([("mamba", m[0], m[1]) for m in mamba_models])
    
    print(f"\nModels to evaluate ({len(all_models)}):")
    current_family = None
    for family, model_name, _ in all_models:
        if family != current_family:
            print(f"  {family.upper()}:")
            current_family = family
        print(f"    - {model_name}")
    
    if args.dry_run:
        print("\n[DRY RUN] Commands that would be executed:")
        for family, model_name, model_id in all_models:
            engine_name = f"hf_{model_id}"
            cmd = f"cd {COGBENCH_PATH} && python3 full_run.py --engine {engine_name}"
            print(f"  ssh {REMOTE_HOST} '{cmd}'  # {family} {model_name}")
        return
    
    # Run evaluations
    print(f"\n{'='*70}")
    print("STARTING COMBINED BATCH EVALUATION")
    print(f"{'='*70}\n")
    
    results = {}
    start_time = datetime.now()
    
    for i, (family, model_name, model_id) in enumerate(all_models):
        print(f"\n[{i+1}/{len(all_models)}] Evaluating {family.upper()} {model_name}")
        
        success = run_model_evaluation(family, model_name, model_id)
        results[f"{family}_{model_name}"] = "completed" if success else "failed"
        
        # Small delay between models
        if i < len(all_models) - 1:
            print(f"\nWaiting 30 seconds before next model...")
            time.sleep(30)
    
    # Sync results at end
    if not args.no_sync:
        sync_results_from_remote(args.local_results)
    
    # Final summary
    elapsed = datetime.now() - start_time
    elapsed_hours = elapsed.total_seconds() / 3600
    
    print(f"\n{'='*70}")
    print("BATCH EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {elapsed_hours:.1f} hours")
    print(f"\nResults:")
    
    current_family = None
    for model_key, status in results.items():
        family, model_name = model_key.split("_", 1)
        if family != current_family:
            print(f"\n{family.upper()}:")
            current_family = family
        print(f"  {model_name:8s}: {status}")
    
    # Count completions
    completed = sum(1 for s in results.values() if "completed" in s)
    failed = len(results) - completed
    
    print(f"\nSummary: {completed} completed, {failed} failed")
    print(f"Results synced to: {args.local_results}")


if __name__ == '__main__':
    main()
