import pandas as pd

print("=== ProbabilisticReasoning - Detailed Analysis ===\n")

for model in ['160m', '410m', '1b', '1.4b']:
    print(f"\n{model.upper()}:")
    df = pd.read_csv(f'results/Experiments/ProbabilisticReasoning/data/hf_EleutherAI/pythia-{model}.csv')
    print(f"  Columns: {list(df.columns)}")
    print(f"  Shape: {df.shape}")
    print(f"  Sample rows:")
    print(df[['informative_lh', 'left_pred', 'red_observation', 'run']].head(10).to_string())
    
    # Check accuracy calculation
    df['accuracy'] = (df['left_pred'] == df['red_observation']).astype(int)
    print(f"\n  Accuracy distribution:")
    print(f"    Total trials: {len(df)}")
    print(f"    Accuracy values: {df['accuracy'].value_counts().to_dict()}")
    print(f"    Mean accuracy: {df['accuracy'].mean():.4f}")
    print(f"    Accuracy per run:")
    per_run = df.groupby('run')['accuracy'].mean()
    print(f"      Mean: {per_run.mean():.4f}, Std: {per_run.std():.4f}")
    print(f"      Min: {per_run.min():.4f}, Max: {per_run.max():.4f}")
    print(f"      Values: {per_run.values}")
