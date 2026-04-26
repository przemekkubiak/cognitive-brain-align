#!/usr/bin/env python3
"""
Setup script for Pythia CogBench evaluation environment.

Handles:
- Installing/verifying Python dependencies
- Cloning CogBench repository if needed
- Checking GPU availability
- Verifying HuggingFace model access

Usage:
    python setup_environment.py [--no_cogbench] [--install_deps]
"""

import argparse
import subprocess
import sys
from pathlib import Path


PYTHON_REQUIREMENTS = [
    "torch>=2.0.0",
    "transformers>=4.30.0",
    "huggingface_hub>=0.16.0",
    "numpy>=1.24.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
]

COGBENCH_GIT = "https://github.com/juliancodaforno/CogBench.git"


def check_python_version():
    """Ensure Python 3.8+."""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or later required")
        return False
    print(f"[OK] Python {sys.version.split()[0]}")
    return True


def check_pip():
    """Verify pip is available."""
    try:
        subprocess.run(["pip", "--version"], capture_output=True, check=True)
        print("[OK] pip available")
        return True
    except subprocess.CalledProcessError:
        print("[ERROR] pip not found")
        return False


def install_requirements(requirements: list):
    """Install Python packages."""
    print("\nInstalling Python dependencies...")
    try:
        cmd = ["pip", "install", "--upgrade"] + requirements
        subprocess.run(cmd, check=True)
        print("[OK] Dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Installation failed: {e}")
        return False


def verify_torch_cuda():
    """Check PyTorch CUDA availability."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            print(f"[OK] CUDA available ({torch.cuda.device_count()} device(s))")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / 1e9
                print(f"    Device {i}: {props.name} ({memory_gb:.1f}GB)")
            return True
        else:
            print("[WARNING] CUDA not available (CPU-only mode)")
            return True  # Not a failure, just a warning
    except Exception as e:
        print(f"[ERROR] Error checking CUDA: {e}")
        return False


def verify_transformers():
    """Verify transformers library can load models."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("[OK] Transformers library functional")
        
        # Try to load a small model metadata (no weights)
        try:
            AutoTokenizer.from_pretrained(
                "EleutherAI/pythia-70m",
                trust_remote_code=True,
            )
            print("[OK] Can access HuggingFace models")
            return True
        except Exception as e:
            print(f"[WARNING] Warning: Could not access HuggingFace: {e}")
            return True  # Not critical, may work on remote machine
    except ImportError as e:
        print(f"[ERROR] Transformers not available: {e}")
        return False


def clone_cogbench(target_dir: Path = Path("./CogBench")):
    """Clone CogBench repository if not present."""
    if target_dir.exists():
        print(f"[OK] CogBench already exists at {target_dir}")
        return True
    
    print(f"\nCloning CogBench to {target_dir}...")
    try:
        subprocess.run(
            ["git", "clone", COGBENCH_GIT, str(target_dir)],
            check=True,
            capture_output=True,
        )
        print(f"[OK] CogBench cloned")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to clone CogBench: {e}")
        return False
    except FileNotFoundError:
        print("[ERROR] git not found")
        return False


def print_summary(success: bool, install_dir: Path):
    """Print setup summary and next steps."""
    print("\n" + "=" * 60)
    print("SETUP SUMMARY")
    print("=" * 60)
    
    if success:
        print("[OK] Environment setup complete!")
        print("\nNext steps:")
        print("1. Update CogBench configuration for HuggingFace models")
        print("2. Run evaluation:")
        print(f"   python evaluate_pythia_cogbench.py --output_dir results/")
        print("\n3. Analyze results:")
        print(f"   python analyze_pythia_results.py --results_dir results/ --plots")
    else:
        print("[ERROR] Setup encountered issues. Please review errors above.")
        print("\nYou may need to:")
        print("- Install system dependencies (CUDA, etc.)")
        print("- Check internet connectivity")
        print("- Verify HuggingFace credentials if needed")
    
    print("=" * 60 + "\n")
    return success


def main():
    parser = argparse.ArgumentParser(
        description="Set up Pythia CogBench evaluation environment"
    )
    
    parser.add_argument(
        "--no_cogbench",
        action="store_true",
        help="Skip CogBench cloning",
    )
    
    parser.add_argument(
        "--install_deps",
        action="store_true",
        help="Install Python dependencies",
    )
    
    parser.add_argument(
        "--cogbench_dir",
        type=Path,
        default=Path("./CogBench"),
        help="Directory for CogBench installation",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PYTHON ENVIRONMENT SETUP")
    print("=" * 60 + "\n")
    
    # Check Python
    if not check_python_version():
        return 1
    
    # Check pip
    if not check_pip():
        return 1
    
    # Install dependencies if requested
    if args.install_deps:
        if not install_requirements(PYTHON_REQUIREMENTS):
            return 1
    
    print()
    
    # Verify installations
    if not verify_torch_cuda():
        return 1
    
    if not verify_transformers():
        return 1
    
    # Clone CogBench if requested
    if not args.no_cogbench:
        if not clone_cogbench(args.cogbench_dir):
            return 1
    
    success = print_summary(True, args.cogbench_dir)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
