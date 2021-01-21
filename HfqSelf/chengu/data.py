import numpy as np
import pandas as pd
import os
import torch
from peewee import *
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline


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

    # 先基础数据预处理，再添加label，后做特征工程
    @staticmethod
    def construct_label_utils(data_basic_processed, transformer_y_pipeline: Pipeline):
        pass

    @staticmethod
    def add_features_utils(data_basic_processed, transformer_x_pipeline: Pipeline):
        pass

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
        return data_raw

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
