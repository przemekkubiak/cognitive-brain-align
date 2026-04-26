#!/usr/bin/env python3
"""
Batch evaluation script for all Pythia models on CogBench.

"""

import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime


PYTHIA_MODELS = [
    ("160m", "EleutherAI/pythia-160m"),
    ("410m", "EleutherAI/pythia-410m"),  # Already done, but included for completeness
    ("1b", "EleutherAI/pythia-1b"),
    ("1.4b", "EleutherAI/pythia-1.4b"),
    ("2.8b", "EleutherAI/pythia-2.8b"),
    ("6.9b", "EleutherAI/pythia-6.9b"),
    ("12b", "EleutherAI/pythia-12b"),
]

# Models to skip (70m is too small and fails)
SKIP_MODELS = {"70m"}

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
        print(f"\n[DONE] {model_name} evaluation completed successfully")
        return True
    except subprocess.TimeoutExpired:
        print(f"\n[FAILED] {model_name} evaluation timed out after 1 hour")
        return False
    except subprocess.CalledProcessError as e:
        print(f"\n[FAILED] {model_name} evaluation failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n[FAILED] {model_name} evaluation error: {e}")
        return False


def check_results(model_name: str) -> int:
    """Check how many experiments completed for a model."""
    cmd = [
        "ssh",
        REMOTE_HOST,
        f"find /root/cognitive-brain-align/CogBench/Experiments -name 'pythia-{model_name}.csv' -type f 2>/dev/null | wc -l"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return int(result.stdout.strip())
    except:
        return 0


def main():
    """Run evaluations for all Pythia models."""
    print("\n" + "="*70)
    print("PYTHIA MODEL BATCH EVALUATION")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    results = {}
    
    for model_name, model_id in PYTHIA_MODELS:
        if model_name in SKIP_MODELS:
            print(f"\nSkipping {model_name} (known to be problematic)")
            continue
        
        # Check if already done
        completed = check_results(model_name)
        if completed >= 6:
            print(f"\n[OK] {model_name} already has {completed} completed experiments, skipping")
            results[model_name] = (True, completed)
            continue
        
        # Run evaluation
        success = run_model_evaluation(model_name, model_id)
        completed = check_results(model_name)
        results[model_name] = (success, completed)
        
        if success and completed >= 6:
            print(f"[DONE] All experiments completed for {model_name}")
        elif completed > 0:
            print(f"[WARNING] Partial completion: {completed} experiments for {model_name}")
        
        # Cool down between evaluations
        if model_name != PYTHIA_MODELS[-1][0]:
            print("\nCooling down before next model (60 seconds)...")
            time.sleep(60)
    
    # Summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    for model_name, (success, completed) in results.items():
        status = "[DONE]" if success and completed >= 6 else "[WARNING]" if completed > 0 else "[FAILED]"
        print(f"{status} {model_name:8} - {completed}/7 experiments completed")
    
    print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    # Return success if all models completed
    all_done = all(completed >= 6 for _, completed in results.values())
    return 0 if all_done else 1


if __name__ == "__main__":
    sys.exit(main())
