import numpy as np
import pandas as pd
import os
import torch
from peewee import *
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
from chengu.transformer import *
from chengu.default_config import default_config


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

    def add_features(self, n_job):
        self.featured_data_lst = Parallel(n_jobs=n_job)(
            delayed(self.add_features_utils)(basic_processed_data)
            for basic_processed_data in self.basic_processed_data_lst
        )
        return self.featured_data_lst

    def process_resample_feature(self, n_job):
        self.resampled_feature_data_lst = Parallel(n_jobs=n_job)(
            delayed(self.process_resample_feature_utils)(featured_data)
            for featured_data in self.featured_data_lst
        )
        return self.resampled_feature_data_lst

    def process_resample_label(self, n_job):
        pass

    @staticmethod
    def screen_feature(data_resampled, train_window_size):
        # 同时完成横向和纵向的筛选：纵向指降采样后的逐一划数据 + 横向指确定使用的特征
        # 此处需要插入特征筛选模块
        # step 1. 完成特征的筛选

        # step 2. 完成X的注意划分样本

        # step 3. 返回np.array, shape:(n_sample, n_feature, n_timestamps)

        pass

    @staticmethod
    def screen_label(data_constructed_label):
        # 完成使用标签的指定，以及逐一划分样本

        pass

    # 该函数既用于完成特征构建后X的降采样，也用于完成label构建后Y（都保留last）的降采样
    # 目的是，保证X和Y的index对齐
    @staticmethod
    def process_resample_feature_utils(
        data_featured, transformer_resample=resample_feature_dict
    ):
        resample_transformer = Resample(
            user_defined_config=default_config, how_dict=transformer_resample
        )
        data_feature_resampled = resample_transformer.transform(data_featured)
        return data_feature_resampled

    @staticmethod
    def process_resample_label_utils(
        data_constructed_label, transformer_resample=resample_label_dict
    ):
        resample_transformer = Resample(
            user_defined_config=default_config, how_dict=transformer_resample
        )
        data_feature_resampled = resample_transformer.transform(data_featured)
        return data_feature_resampled

    # 先基础数据预处理，再添加label，后做特征工程
    @staticmethod
    def construct_label_utils(data_basic_processed, transformer_y_pipeline: Pipeline):
        return

    @staticmethod
    def add_features_utils(data_basic_processed, transformer_x_pipeline: Pipeline):
        return

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
