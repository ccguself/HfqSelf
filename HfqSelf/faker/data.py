import numpy as np
import pandas as pd
import os
import torch
from peewee import *
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline


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


def mark_price_limits(x):
    if (x["bid_price_1"] != 0) & (x["ask_price_1"] != 0):
        return np.nan
    else:
        return x["mid_price"]


class RawData:
    def __init__(
        self,
        db_path=r"D:\Trader\.vntrader\database.db",
        symbol="rb2105",
        trade_date_lst=["2021-01-18", "2021-01-19"],
    ):
        self.db_path = db_path
        self.symbol = symbol
        self.trade_date_lst = trade_date_lst

    def fetch_data(self, n_job):
        # 从数据库中提取数据
        self.raw_data_lst = Parallel(n_jobs=n_job)(
            delayed(self.fetch_data_utils)(self.db_path, self.symbol, trade_date)
            for trade_date in self.trade_date_lst
        )
        return self.raw_data_lst

    def basic_process_data(self, n_job):
        # 数据的基础预处理，截取需要使用的列
        self.basic_processed_data_lst = Parallel(n_jobs=n_job)(
            delayed(self.basic_process_data_utils)(raw_data)
            for raw_data in self.raw_data_lst
        )
        return self.basic_processed_data_lst
    
    def generate_predict_timestamp(self, mode="simple"):
        if mode == "simple":
            


    @staticmethod
    def fetch_data_utils(file_path, symbol, trade_date):
        db = SqliteDatabase(file_path)
        df_market = pd.read_sql(
            "select * from dbtickdata where (symbol='%s') and Date(datetime) between '%s' and '%s'"
            % (symbol, trade_date, trade_date),
            db,
            parse_dates="datetime",
        )
        return df_market

    @staticmethod
    def basic_process_data_utils(data_raw):
        keep_columns = [
            "datetime",
            "last_price",
            "volume",
            "bid_price_1",
            "bid_volume_1",
            "ask_price_1",
            "ask_volume_1",
            "open_interest",
        ]
        data_raw = data_raw[keep_columns]
        data_raw["mid_price"] = (data_raw["bid_price_1"] + data_raw["ask_price_1"]) / 2
        data_raw["mid_price"] = data_raw.apply(mark_price_limits)

        # step2. 对volume, open interest对差值处理
        data_raw["delta_volume"] = data_raw["volume"].diff(1).fillna(0)
        data_raw["delta_oi"] = data_raw["open_interest"].diff(1).fillna(0)

        # step3. 对交易类别做判断
        data_raw["tick_trade_type"] = data_raw.apply(classify_trade_type, axis=1)
        trade_types_lst = ["BO", "BC", "LO", "SC", "LE", "SO", "LC", "SE"]
        for i, trade_type in enumerate(trade_types_lst):
            data_raw[trade_type] = np.where(
                data_raw["tick_trade_type"] == trade_type, data_raw["delta_oi"], 0
            )
        # 只保留需要使用的列
        drop_columns = ["tick_trade_type", "volume", "open_interest"]
        return data_raw.drop(drop_columns, axis=1)
