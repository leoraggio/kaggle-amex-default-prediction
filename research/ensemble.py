from pathlib import Path

import pandas as pd

project_root = Path(__file__).parents[1]

def ensemble() -> pd.DataFrame:
    seed42 = pd.read_csv(project_root / "data/Predictions/test_lgbm_dart_baseline_5fold_seed42.csv")
    seed2022 = pd.read_csv(
        project_root / "data/Predictions/test_lgbm_dart_baseline_5fold_seed2022.csv"
    )
    predictions = seed42.set_index("customer_ID").join(
        seed2022.set_index("customer_ID"), rsuffix="seed2022"
    )
    predictions = predictions.mean(axis=1).reset_index()
    predictions.columns = ["customer_ID", "prediction"]
    return predictions


if __name__ == "__main__":
    new_predictions = ensemble()
    new_predictions.to_csv(
        project_root / "data/Predictions/test_lgbm_dart_baseline_5fold_ensemble.csv", index=False
    )
