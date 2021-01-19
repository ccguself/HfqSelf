import os
import numpy as np
import pandas as pd
import seaborn as sns
import datetime
from joblib import Parallel, delayed
from peewee import *


class SingleCombinedLog:
    def __init__(self, df_market_interval):
        self.df_market_interval = df_market_interval








class MarketLog:
    def __init__(self, db_path, datetime_start, datetime_end, symbol):
        self.db_path = db_path
        self.datetime_start = datetime_start
        self.datetime_end = datetime_end
        self.symbol = symbol
        self._fetch_data()

    def _fetch_data(self):
        # step1. 从数据库中提取数据
        db = SqliteDatabase(self.db_path)
        self.df_market = pd.read_sql(
            "select * from dbtickdata where (symbol='%s') and Date(datetime) between '%s' and '%s'"
            % (self.symbol, self.datetime_start, self.datetime_end),
            db,
            parse_dates="datetime",
        )

        # step2. 数据预处理
        keep_columns = [
            "datetime",
            "last_price",
            "volume",
            "bid_price_1",
            "bid_volume_1",
            "ask_price_1",
            "ask_volume_1",
        ]
        self.df_market = self.df_market[keep_columns]
        self.df_market["datetime"] = self.df_market["datetime"].apply(
            lambda x: x.timestamp()
        )
        return self.df_market


class AnalysisLog:
    def __init__(self, total_market_log:MarketLog, start_interval=2000):
        self.total_market_log = total_market_log
        self.start_interval = start_interval
        self.dict_long = {}  # datetime: SingleCombinedLog
        self.dict_short = {}  # datetime: SingleCombinedLog
    
    def generate_single_combined_log(self):
        



