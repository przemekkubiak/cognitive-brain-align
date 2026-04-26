import pandas as pd
import numpy as np

# Check HorizonTask structure
df = pd.read_csv('results/Experiments/HorizonTask/data/hf_EleutherAI/pythia-160m.csv')
print("HorizonTask (160m):")
print(f"  Shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")
print(f"  Unique runs: {df['run'].nunique()}")
print(f"  Trials per run (first 5):")
for run in sorted(df['run'].unique())[:5]:
    count = len(df[df['run'] == run])
    print(f"    Run {run}: {count} trials")
print(f"\n  reward column stats:")
print(f"    Min: {df['reward'].min()}, Max: {df['reward'].max()}")
print(f"    Mean: {df['reward'].mean():.4f}")
print(f"  First few rows:")
print(df.head())

# Check RestlessBandit
print("\n" + "="*50)
df2 = pd.read_csv('results/Experiments/RestlessBandit/data/hf_EleutherAI/pythia-410m.csv')
print("RestlessBandit (410m):")
print(f"  Shape: {df2.shape}")
print(f"  Columns: {list(df2.columns)}")
print(f"  Unique runs: {df2['run'].nunique()}")
print(f"  Reward stats:")
print(f"    Min: {df2['reward'].min()}, Max: {df2['reward'].max()}")
print(f"    Mean: {df2['reward'].mean():.4f}")
print(f"  First few rows:")
print(df2.head())
