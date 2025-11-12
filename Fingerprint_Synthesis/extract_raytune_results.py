import os
import json
import pandas as pd
from ray import tune
from ray.tune import ExperimentAnalysis

# ---- CHANGE THIS ----
# Path to your Ray Tune experiment directory
EXPERIMENT_DIR = "/home/jovyan/work/runs/ray_tune/gan_tuning"

# Metric to optimize and its mode
METRIC = "loss_G"
MODE = "min"

# ----------------------

def extract_raytune_results(experiment_dir, metric, mode):
    print(f"Loading Ray Tune experiment from: {experiment_dir}")
    analysis = ExperimentAnalysis(experiment_dir)

    # --- 1. Save all trial results to CSV ---
    df = analysis.dataframe(metric=metric, mode=mode)
    csv_path = os.path.join(experiment_dir, "tuning_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved all trial results → {csv_path}")

    # --- 2. Extract best configuration ---
    best_config = analysis.get_best_config(metric=metric, mode=mode)
    config_path = os.path.join(experiment_dir, "best_config.json")
    with open(config_path, "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"✅ Saved best config → {config_path}")

    # --- 3. Extract best checkpoint directory ---
    try:
        best_checkpoint = analysis.get_best_checkpoint(
            trial=analysis.get_best_trial(metric=metric, mode=mode),
            metric=metric,
            mode=mode
        )
        if best_checkpoint:
            print(f"✅ Best checkpoint path → {best_checkpoint.path}")
        else:
            print("⚠️ No checkpoints found for best trial.")
    except Exception as e:
        print(f"⚠️ Could not retrieve best checkpoint: {e}")

    print("Done.")

if __name__ == "__main__":
    extract_raytune_results(EXPERIMENT_DIR, METRIC, MODE)
