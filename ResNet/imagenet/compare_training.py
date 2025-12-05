#!/usr/bin/env python3
"""
Parallel Training Script for ResNet Comparison on ImageNet

This script trains two ResNet models in parallel:
1. Baseline ResNet (without Stochastic Depth)
2. ResNet with Stochastic Depth

Designed for systems with sufficient hardware resources (multi-GPU or multiple machines).
"""

import argparse
import subprocess
import sys
import os
from datetime import datetime
import json


def create_training_config(base_dir, model_name, sd_enabled, args):
    """Create training configuration for a model."""
    output_dir = os.path.join(base_dir, model_name)
    
    cmd = [
        sys.executable,
        "full_imagenet_train.py",
        "--data_dir", args.data_dir,
        "--out", output_dir,
        "--depth", str(args.depth),
        "--sd", "1" if sd_enabled else "0",
        "--epochs", str(args.epochs),
        "--batch", str(args.batch),
        "--mixup", str(args.mixup),
        "--label_smoothing", str(args.label_smoothing),
        "--seed", str(args.seed)
    ]
    
    if sd_enabled:
        cmd.extend(["--pL", str(args.pL)])
    
    if args.no_mixed_precision:
        cmd.append("--no_mixed_precision")
    
    if args.cache:
        cmd.append("--cache")
    
    return cmd, output_dir


def run_sequential(args):
    """Run training sequentially (one after another)."""
    print("="*80)
    print("SEQUENTIAL TRAINING MODE")
    print("="*80)
    print("\nThis will train two models one after another:")
    print("  1. Baseline ResNet (no Stochastic Depth)")
    print("  2. ResNet with Stochastic Depth")
    print()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(args.output_base, f"comparison_{timestamp}")
    os.makedirs(base_dir, exist_ok=True)
    
    # Train baseline first
    print("\n" + "="*80)
    print("STEP 1/2: Training Baseline Model (No Stochastic Depth)")
    print("="*80 + "\n")
    
    baseline_cmd, baseline_dir = create_training_config(
        base_dir, "baseline_no_sd", False, args
    )
    
    print(f"Command: {' '.join(baseline_cmd)}\n")
    result1 = subprocess.run(baseline_cmd)
    
    if result1.returncode != 0:
        print(f"\n❌ Baseline training failed with exit code {result1.returncode}")
        return False
    
    print("\n✓ Baseline training completed successfully!\n")
    
    # Train with stochastic depth
    print("\n" + "="*80)
    print("STEP 2/2: Training Model with Stochastic Depth")
    print("="*80 + "\n")
    
    sd_cmd, sd_dir = create_training_config(
        base_dir, f"stochastic_depth_pL{args.pL}", True, args
    )
    
    print(f"Command: {' '.join(sd_cmd)}\n")
    result2 = subprocess.run(sd_cmd)
    
    if result2.returncode != 0:
        print(f"\n❌ Stochastic Depth training failed with exit code {result2.returncode}")
        return False
    
    print("\n✓ Stochastic Depth training completed successfully!\n")
    
    # Print summary
    print("\n" + "="*80)
    print("TRAINING COMPARISON COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {base_dir}/")
    print(f"  - Baseline: {baseline_dir}/")
    print(f"  - Stochastic Depth: {sd_dir}/")
    print("\nTo compare results, check the training_summary.csv files in each directory.")
    
    return True


def run_parallel(args):
    """Run training in parallel using separate processes."""
    print("="*80)
    print("PARALLEL TRAINING MODE")
    print("="*80)
    print("\nThis will train two models simultaneously:")
    print("  1. Baseline ResNet (no Stochastic Depth)")
    print("  2. ResNet with Stochastic Depth")
    print("\n⚠️  WARNING: This requires substantial GPU resources!")
    print("   Recommended: 2 GPUs with at least 32GB VRAM each")
    print()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(args.output_base, f"comparison_{timestamp}")
    os.makedirs(base_dir, exist_ok=True)
    
    # Prepare both commands
    baseline_cmd, baseline_dir = create_training_config(
        base_dir, "baseline_no_sd", False, args
    )
    
    sd_cmd, sd_dir = create_training_config(
        base_dir, f"stochastic_depth_pL{args.pL}", True, args
    )
    
    print("\nStarting parallel training processes...")
    print(f"Baseline command: {' '.join(baseline_cmd)}")
    print(f"Stochastic Depth command: {' '.join(sd_cmd)}")
    print()
    
    # Set up GPU assignment if available
    # Process 1: GPU 0, Process 2: GPU 1
    env1 = os.environ.copy()
    env2 = os.environ.copy()
    
    if args.gpu_split:
        env1['CUDA_VISIBLE_DEVICES'] = '0'
        env2['CUDA_VISIBLE_DEVICES'] = '1'
        print("✓ GPU assignment: Baseline -> GPU:0, Stochastic Depth -> GPU:1\n")
    else:
        print("ℹ Using default GPU assignment (TensorFlow will manage)\n")
    
    # Start both processes
    process1 = subprocess.Popen(
        baseline_cmd,
        env=env1,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    process2 = subprocess.Popen(
        sd_cmd,
        env=env2,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    print("✓ Both training processes started!\n")
    print("Waiting for completion...")
    print("(This may take several days depending on your hardware)\n")
    
    # Save logs
    log1_path = os.path.join(baseline_dir, "training_output.log")
    log2_path = os.path.join(sd_dir, "training_output.log")
    
    os.makedirs(baseline_dir, exist_ok=True)
    os.makedirs(sd_dir, exist_ok=True)
    
    with open(log1_path, 'w') as f1, open(log2_path, 'w') as f2:
        # Stream outputs
        while True:
            out1 = process1.stdout.readline()
            out2 = process2.stdout.readline()
            
            if out1:
                f1.write(out1)
                f1.flush()
            if out2:
                f2.write(out2)
                f2.flush()
            
            # Check if both processes finished
            if process1.poll() is not None and process2.poll() is not None:
                break
    
    # Get return codes
    ret1 = process1.returncode
    ret2 = process2.returncode
    
    print("\n" + "="*80)
    print("PARALLEL TRAINING COMPLETE")
    print("="*80)
    
    if ret1 == 0 and ret2 == 0:
        print("\n✓ Both models trained successfully!")
    elif ret1 != 0:
        print(f"\n❌ Baseline training failed (exit code: {ret1})")
    elif ret2 != 0:
        print(f"\n❌ Stochastic Depth training failed (exit code: {ret2})")
    
    print(f"\nResults saved to: {base_dir}/")
    print(f"  - Baseline: {baseline_dir}/")
    print(f"  - Stochastic Depth: {sd_dir}/")
    print(f"\nTraining logs:")
    print(f"  - Baseline: {log1_path}")
    print(f"  - Stochastic Depth: {log2_path}")
    
    return ret1 == 0 and ret2 == 0


def main():
    parser = argparse.ArgumentParser(
        description='Train and compare ResNet models with and without Stochastic Depth',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sequential training (one after another)
  python compare_training.py --data_dir /data/imagenet --mode sequential
  
  # Parallel training (simultaneous, requires 2 GPUs)
  python compare_training.py --data_dir /data/imagenet --mode parallel --gpu_split
  
  # Custom configuration
  python compare_training.py --data_dir /data/imagenet --depth 101 --batch 512 \\
      --epochs 120 --mixup 0.2 --label_smoothing 0.1
        """
    )
    
    # Required
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to ImageNet dataset')
    
    # Training mode
    parser.add_argument('--mode', type=str, default='sequential',
                       choices=['sequential', 'parallel'],
                       help='Training mode: sequential or parallel (default: sequential)')
    
    # Model configuration
    parser.add_argument('--depth', type=int, default=50, choices=[50, 101, 152],
                       help='ResNet depth (default: 50)')
    parser.add_argument('--pL', type=float, default=0.5,
                       help='Final survival probability for SD (default: 0.5)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=120,
                       help='Number of epochs (default: 120)')
    parser.add_argument('--batch', type=int, default=256,
                       help='Batch size (default: 256)')
    parser.add_argument('--mixup', type=float, default=0.2,
                       help='MixUp alpha (default: 0.2)')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                       help='Label smoothing (default: 0.1)')
    
    # Output
    parser.add_argument('--output_base', type=str, default='./comparison_results',
                       help='Base output directory (default: ./comparison_results)')
    
    # Options
    parser.add_argument('--gpu_split', action='store_true',
                       help='Split models across GPUs in parallel mode (GPU:0 and GPU:1)')
    parser.add_argument('--no_mixed_precision', action='store_true',
                       help='Disable mixed precision training')
    parser.add_argument('--cache', action='store_true',
                       help='Cache dataset in memory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Print configuration
    print("\n" + "="*80)
    print("RESNET COMPARISON TRAINING")
    print("="*80)
    print("\nConfiguration:")
    print(f"  Dataset: {args.data_dir}")
    print(f"  Mode: {args.mode}")
    print(f"  ResNet Depth: {args.depth}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch}")
    print(f"  MixUp: {args.mixup}")
    print(f"  Label Smoothing: {args.label_smoothing}")
    print(f"  Stochastic Depth pL: {args.pL}")
    print(f"  Output: {args.output_base}")
    print()
    
    # Check data directory
    if not os.path.isdir(args.data_dir):
        print(f"❌ Error: Data directory not found: {args.data_dir}")
        return 1
    
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    
    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        print(f"❌ Error: Expected 'train' and 'val' subdirectories in {args.data_dir}")
        return 1
    
    # Run training
    if args.mode == 'sequential':
        success = run_sequential(args)
    else:
        success = run_parallel(args)
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
