import numpy as np
import pandas as pd
import torch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


class AddBasicFeatures(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        basic_feature_lst=["mid_price", "delta_volume", "delta_oi", "tick_trade_type"],
    ):
        self.basic_feature_lst = basic_feature_lst

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        # step1. 添加mid_price
        X["mid_price"] = (X["bid_price_1"] + X["ask_price_1"]) / 2

        # step2. 对volume, open interest对差值处理
        X["delta_volume"] = X["volume"].diff(1).fillna(0)
        X["delta_oi"] = X["open_interest"].diff(1).fillna(0)

        # step3. 对交易类别做判断
        X["tick_trade_type"] = X.apply(self.classify_trade_type, axis=1)
        trade_types_lst = ["BO", "BC", "LO", "SC", "LE", "SO", "LC", "SE"]
        for i, trade_type in enumerate(trade_types_lst):
            X[trade_type] = np.where(
                X["tick_trade_type"] == trade_type, X["delta_oi"], 0
            )

        # 只保留需要使用的列
        drop_columns = ["tick_trade_type", "volume", "open_interest"]
        return X.drop(drop_columns, axis=1)

    @staticmethod
    def classify_trade_type(x):
        if x["delta_volume"] == abs(x["delta_oi"]):
            if x["delta_oi"] > 0:
                return "BO"  # 双开
            else:
                return "BC"  # 双平
        else:
            if x["last_price"] == x["ask_price_1"]:
                if x["delta_oi"] > 0:
                    return "LO"  # 多开
                elif x["delta_oi"] < 0:
                    return "SC"  # 空平
                else:
                    return "LE"  # 多换
            else:
                if x["delta_oi"] > 0:
                    return "SO"  # 空开
                elif x["delta_oi"] < 0:
                    return "LC"  # 多平
                else:
                    return "SE"  # 空换


class FilterInvalidSample(BaseEstimator, TransformerMixin):
    def __init__(self, process_situation_lst=["price_limits"]):
        self.process_situation_lst = process_situation_lst

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        # 剔除涨跌停
        X = X.loc[(X["bid_price_1"] != 0) & (X["ask_price_1"] != 0)]
        return X


class GetLabel(BaseEstimator, TransformerMixin):
    def __init__(self, user_defined_config):
        self.user_defined_config = user_defined_config

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        # step1. 完成计算（列值）
        label_df = X.copy()
        label_df["mid_price"] = (label_df["bid_price_1"] + label_df["ask_price_1"]) / 2
        label_df["delta_volume"] = label_df["volume"].diff(1).fillna(0)
        label_df["delta_oi"] = label_df["open_interest"].diff(1).fillna(0)

        # 注意，此处要对标签有问题的做NA，用于后面剔除这些样本！！！
        label_df["mid_price"] = label_df.apply(self.mark_price_limits)

        # 保留需要使用的列
        keep_columns = ["datetime", "mid_price"]
        label_df = label_df[keep_columns]

        # step2. 根据default_config完成shift

        # step3. set_index为后面的降采样做准备
        return label_df

    @staticmethod
    def mark_price_limits(x):
        if (x["bid_price_1"] != 0) & (x["ask_price_1"] != 0):
            return np.nan
        else:
            return x["mid_price"]


class Resample(BaseEstimator, TransformerMixin):
    def __init__(self, user_defined_config, how_dict):
        self.user_defined_config = user_defined_config
        self.how_dict = how_dict

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_resampled = X.resample(
            rule=self.user_defined_config["size_resample"],
            label="right",
            closed="right",
        ).agg(self.how_dict)
        X_resampled = X_resampled.dropna(how="any")
        return X_resampled


# 将标准化，归一化，去极值等操作，分为算子计算和处理两个部分
class SliceData(BaseEstimator, TransformerMixin):
    def __init__(self, user_defined_config):
        self.user_defined_config = user_defined_config

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        X_tensor = torch.from_numpy(X.values)
        if self.user_defined_config["slice_overlap"] == False:
            remainder = X.shape[0] % self.user_defined_config["size_train"]
            input_tensor = X_tensor[
                (self.user_defined_config["size_train"] + remainder) :, :
            ]
            return input_tensor.view(
                -1, self.user_defined_config["size_train"], input_tensor.shape[1]
            )
        else:
            input_tensor = torch.zero(
                self.user_defined_config["size_train"], X.shape[1]
            )
            for i in range(
                0,
                X.shape[0] - self.user_defined_config["size_train"],
                self.user_defined_config["size_step_increment"],
            ):
                temp_tensor = X.iloc[
                    i : i + self.user_defined_config["size_train"],
                ]
                input_tensor = torch.stack((input_tensor, temp_tensor), 0)

            return input_tensor


resample_feature_dict = {
    "last_price": "last",
    "volume": "last",
    "bid_price_1": "last",
    "bid_volume_1": "last",
    "ask_price_1": "last",
    "ask_volume_1": "last",
}

resample_label_dict = {
    "last_price": "last",
    "volume": "last",
    "bid_price_1": "last",
    "bid_volume_1": "last",
    "ask_price_1": "last",
    "ask_volume_1": "last",
}
