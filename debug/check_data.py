import pandas as pd

print("=== HorizonTask ===")
for model in ['160m', '410m', '1b']:
    df = pd.read_csv(f'results/Experiments/HorizonTask/data/hf_EleutherAI/pythia-{model}.csv')
    print(f"\n{model}:")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Reward range: {df['reward'].min():.1f} - {df['reward'].max():.1f}")
    print(f"  Mean reward (all): {df['reward'].mean():.3f}")
    # Check per-run means
    per_run = df.groupby('run')['reward'].mean()
    print(f"  Per-run mean: {per_run.mean():.3f} ± {per_run.std():.3f}")
    print(f"  Sample data:")
    print(df[['trial', 'reward', 'run']].head(10))
    break

print("\n=== RestlessBandit ===")
for model in ['160m', '410m', '1b']:
    df = pd.read_csv(f'results/Experiments/RestlessBandit/data/hf_EleutherAI/pythia-{model}.csv')
    print(f"\n{model}:")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Reward range: {df['reward'].min():.1f} - {df['reward'].max():.1f}")
    print(f"  Mean reward (all): {df['reward'].mean():.3f}")
    # Check per-run means
    per_run = df.groupby('run')['reward'].mean()
    print(f"  Per-run mean: {per_run.mean():.3f} ± {per_run.std():.3f}")
    print(f"  Sample data:")
    print(df[['trial', 'reward', 'run']].head(10))
    break
