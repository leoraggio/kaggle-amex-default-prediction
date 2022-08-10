import os
from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from research.utils import create_logger

logger = create_logger("dataprep")

BUCKET = "gs://leoraggio-kaggle"
COMPETITION = "amex-default-prediction"


class DataBunch:
    def __init__(
        self,
        train_data: pd.DataFrame,
        train_labels: pd.DataFrame,
        test_data: pd.DataFrame,
    ) -> None:
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data

        self._features = None
        self._num_features = None

        self.cat_features = [
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

    @property
    def features(self) -> List[str]:
        if not self._features:
            self._features = self.train_data.drop(
                columns=["customer_ID", "S_2"]
            ).columns.to_list()
        return self._features

    @property
    def num_features(self) -> List[str]:
        if not self._num_features:
            self._num_features = [
                col for col in self.features if col not in self.cat_features
            ]
        return self._num_features


def get_difference(data, num_features):
    df1 = []
    customer_ids = []
    for customer_id, df in tqdm(data.groupby(["customer_ID"])):
        # Get the differences
        diff_df1 = df[num_features].diff(1).iloc[[-1]].values.astype(np.float32)
        # Append to lists
        df1.append(diff_df1)
        customer_ids.append(customer_id)
    # Concatenate
    df1 = np.concatenate(df1, axis=0)
    # Transform to dataframe
    df1 = pd.DataFrame(
        df1, columns=[col + "_diff1" for col in data[num_features].columns]
    )
    # Add customer id
    df1["customer_ID"] = customer_ids
    return df1


def create_agg_features(
    data: pd.DataFrame, num_features: List[str], cat_features: List[str]
) -> pd.DataFrame:
    # Aggregate numerical features
    train_num_agg = data.groupby("customer_ID")[num_features].agg(
        ["mean", "std", "min", "max", "last"]
    )
    train_num_agg.columns = ["_".join(x) for x in train_num_agg.columns]
    train_num_agg.reset_index(inplace=True)

    # Aggregate categorical features
    train_cat_agg = data.groupby("customer_ID")[cat_features].agg(
        ["count", "last", "nunique"]
    )
    train_cat_agg.columns = ["_".join(x) for x in train_cat_agg.columns]
    train_cat_agg.reset_index(inplace=True)
    return train_num_agg.merge(train_cat_agg, how="inner", on="customer_ID")


def preprocess_data(data_bunch: DataBunch) -> None:
    logger.info("Starting feature engineer...")

    logger.info("Creating aggregate features on training data...")
    train = create_agg_features(
        data=data_bunch.train_data,
        num_features=data_bunch.num_features,
        cat_features=data_bunch.cat_features,
    )

    logger.info("Getting differences on training data...")
    train = train.merge(
        get_difference(data_bunch.train_data, data_bunch.num_features),
        how="inner",
        on="customer_ID",
    ).merge(data_bunch.train_labels, how="inner", on="customer_ID")

    # Feature engineer on test data
    logger.info("Creating aggregate features on test data...")
    test = create_agg_features(
        data=data_bunch.test_data,
        num_features=data_bunch.num_features,
        cat_features=data_bunch.cat_features,
    )

    logger.info("Getting differences on test data...")
    test = test.merge(
        get_difference(data_bunch.test_data, data_bunch.num_features),
        how="inner",
        on="customer_ID",
    )

    # Handle categorical features
    logger.info("Encoding categorical features...")
    cat_features = [f"{cf}_last" for cf in data_bunch.cat_features]
    for cat_col in cat_features:
        encoder = LabelEncoder()
        train[cat_col] = encoder.fit_transform(train[cat_col])
        test[cat_col] = encoder.transform(test[cat_col])

    # Round last float features to 2 decimal place
    logger.info("Creating rounded features...")
    num_cols = list(
        train.dtypes[(train.dtypes == "float32") | (train.dtypes == "float64")].index
    )
    num_cols = [col for col in num_cols if "last" in col]
    for col in num_cols:
        train[col + "_round2"] = train[col].round(2)
        test[col + "_round2"] = test[col].round(2)

    # Get the difference between last and mean
    logger.info("Getting the difference between last and mean features...")
    num_cols = [col for col in train.columns if "last" in col]
    num_cols = [col[:-5] for col in num_cols if "round" not in col]
    for col in num_cols:
        try:
            train[f"{col}_last_mean_diff"] = train[f"{col}_last"] - train[f"{col}_mean"]
            test[f"{col}_last_mean_diff"] = test[f"{col}_last"] - test[f"{col}_mean"]
        except:
            pass

    # Save files
    logger.info("Saving train_data...")
    train.to_parquet(
        "gs://leoraggio-kaggle/amex-default-prediction/data/processed/train_data.parquet"
    )
    logger.info("Saving test data")
    test.to_parquet(
        "gs://leoraggio-kaggle/amex-default-prediction/data/processed/test_data.parquet"
    )


if __name__ == "__main__":
    logger.info("Loading train data...")
    train_data = pd.read_parquet(
        os.path.join(BUCKET, COMPETITION, "data/optimized/train.parquet")
    )
    train_labels = pd.read_csv(
        os.path.join(BUCKET, COMPETITION, "data/raw/train_labels.csv")
    )
    logger.info("Loading test data...")
    test_data = pd.read_parquet(
        os.path.join(BUCKET, COMPETITION, "data/optimized/test.parquet")
    )

    data_bunch = DataBunch(train_data, train_labels, test_data)
    preprocess_data(data_bunch)
