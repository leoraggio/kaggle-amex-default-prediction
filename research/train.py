import gc
import os
import warnings

from metric import amex_metric, lgb_amex_metric

warnings.filterwarnings("ignore")
import random

import joblib
import numpy as np
import pandas as pd
import scipy as sp

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm


class CFG:
    input_dir = "/content/data/"
    seed = 2022
    n_folds = 5
    target = "target"
    boosting_type = "dart"
    metric = "binary_logloss"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def train_and_evaluate(train, test):
    # Label encode categorical features
    cat_features = [
        "B_30",
        "B_38",
        "D_114",
        "D_116",
        "D_117",
        "D_120",
        "D_126",
        "D_63",
        "D_64",
        "D_66",
        "D_68",
    ]
    cat_features = [f"{cf}_last" for cf in cat_features]
    for cat_col in cat_features:
        encoder = LabelEncoder()
        train[cat_col] = encoder.fit_transform(train[cat_col])
        test[cat_col] = encoder.transform(test[cat_col])
    # Round last float features to 2 decimal place
    num_cols = list(
        train.dtypes[(train.dtypes == "float32") | (train.dtypes == "float64")].index
    )
    num_cols = [col for col in num_cols if "last" in col]
    for col in num_cols:
        train[col + "_round2"] = train[col].round(2)
        test[col + "_round2"] = test[col].round(2)
    # Get the difference between last and mean
    num_cols = [col for col in train.columns if "last" in col]
    num_cols = [col[:-5] for col in num_cols if "round" not in col]
    for col in num_cols:
        try:
            train[f"{col}_last_mean_diff"] = train[f"{col}_last"] - train[f"{col}_mean"]
            test[f"{col}_last_mean_diff"] = test[f"{col}_last"] - test[f"{col}_mean"]
        except:
            pass
    # Transform float64 and float32 to float16
    num_cols = list(
        train.dtypes[(train.dtypes == "float32") | (train.dtypes == "float64")].index
    )
    for col in tqdm(num_cols):
        train[col] = train[col].astype(np.float16)
        test[col] = test[col].astype(np.float16)
    # Get feature list
    features = [col for col in train.columns if col not in ["customer_ID", CFG.target]]
    params = {
        "objective": "binary",
        "metric": CFG.metric,
        "boosting": CFG.boosting_type,
        "seed": CFG.seed,
        "num_leaves": 100,
        "learning_rate": 0.01,
        "feature_fraction": 0.20,
        "bagging_freq": 10,
        "bagging_fraction": 0.50,
        "n_jobs": -1,
        "lambda_l2": 2,
        "min_data_in_leaf": 40,
    }
    # Create a numpy array to store test predictions
    test_predictions = np.zeros(len(test))
    # Create a numpy array to store out of folds predictions
    oof_predictions = np.zeros(len(train))
    kfold = StratifiedKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)
    for fold, (trn_ind, val_ind) in enumerate(kfold.split(train, train[CFG.target])):
        print(" ")
        print("-" * 50)
        print(f"Training fold {fold} with {len(features)} features...")
        x_train, x_val = train[features].iloc[trn_ind], train[features].iloc[val_ind]
        y_train, y_val = (
            train[CFG.target].iloc[trn_ind],
            train[CFG.target].iloc[val_ind],
        )
        lgb_train = lgb.Dataset(x_train, y_train, categorical_feature=cat_features)  # type: ignore
        lgb_valid = lgb.Dataset(x_val, y_val, categorical_feature=cat_features)  # type: ignore
        model = lgb.train(
            params=params,
            train_set=lgb_train,
            num_boost_round=10500,
            valid_sets=[lgb_train, lgb_valid],
            early_stopping_rounds=1500,
            verbose_eval=500,
            feval=lgb_amex_metric,
        )
        # Save best model
        joblib.dump(
            model,
            f"../Models/lgbm_{CFG.boosting_type}_fold{fold}_seed{CFG.seed}.pkl",
        )
        # Predict validation
        val_pred = model.predict(x_val)
        # Add to out of folds array
        oof_predictions[val_ind] = val_pred
        # Predict the test set
        test_pred = model.predict(test[features])
        test_predictions += test_pred / CFG.n_folds
        # Compute fold metric
        score = amex_metric(y_val, val_pred)
        print(f"Our fold {fold} CV score is {score}")
        del x_train, x_val, y_train, y_val, lgb_train, lgb_valid
        gc.collect()
    # Compute out of folds metric
    score = amex_metric(train[CFG.target], oof_predictions)
    print(f"Our out of folds CV score is {score}")
    # Create a dataframe to store out of folds predictions
    oof_df = pd.DataFrame(
        {
            "customer_ID": train["customer_ID"],
            "target": train[CFG.target],
            "prediction": oof_predictions,
        }
    )
    oof_df.to_csv(
        f"../data/OOF/oof_lgbm_{CFG.boosting_type}_baseline_{CFG.n_folds}fold_seed{CFG.seed}.csv",
        index=False,
    )
    # Create a dataframe to store test prediction
    test_df = pd.DataFrame(
        {"customer_ID": test["customer_ID"], "prediction": test_predictions}
    )
    test_df.to_csv(
        f"../data/Predictions/test_lgbm_{CFG.boosting_type}_baseline_{CFG.n_folds}fold_seed{CFG.seed}.csv",
        index=False,
    )
