from pathlib import Path

import pandas as pd

project_root = Path(__file__).parents[1]


def ensemble_lgbm() -> pd.DataFrame:
    seed42 = pd.read_csv(
        project_root / "data/Predictions/test_lgbm_dart_baseline_5fold_seed42.csv"
    )
    seed2022 = pd.read_csv(
        project_root / "data/Predictions/test_lgbm_dart_baseline_5fold_seed2022.csv"
    )
    predictions = seed42.set_index("customer_ID").join(
        seed2022.set_index("customer_ID"), rsuffix="seed2022"
    )
    predictions = predictions.mean(axis=1).reset_index()
    predictions.columns = ["customer_ID", "prediction"]
    return predictions


def ensemble() -> pd.DataFrame:
    lgbm_score = 0.799
    rf_score = 0.77
    lgbm_weight = lgbm_score / (lgbm_score + rf_score)
    rf_weight = rf_score / (lgbm_score + rf_score)

    lgbm_predictions = ensemble_lgbm()
    rf_predictions = pd.read_csv(
        project_root / "data/Predictions/test_rf_baseline_5fold_seed2022.csv"
    )

    final_predictions = lgbm_predictions.merge(
        rf_predictions, on="customer_ID", how="inner", suffixes=("_lgbm", "_rf")
    )
    final_predictions["prediction"] = (
        final_predictions["prediction_lgbm"] * lgbm_weight
        + final_predictions["prediction_rf"] * rf_weight
    )
    return final_predictions[["customer_ID", "prediction"]]


if __name__ == "__main__":
    new_predictions = ensemble()
    new_predictions.to_csv(
        project_root / "data/Predictions/test_ensemble.csv", index=False
    )
