from metrics import compute_pi_subsampling_distribution
from metrics import plot_pi_distribution

root_dir = r"D:\tester\MMA_Patient5\5"

result = compute_pi_subsampling_distribution(
    csv_path=f"{root_dir}\predictions.csv",
    subsample_size=1000,
    n_iterations=3000,
    random_seed=67,
)

plot_pi_distribution(result, save_path=f"{root_dir}\\pi_distribution_1000samplesize.png")

print(f"PI = {result['point_estimate_pi']:.2f}% ± {1.96*result['std_pi']:.2f} (95% CI)")