import os
import numpy as np
import pandas as pd
import seaborn as sns
import datetime
from joblib import Parallel, delayed


class Log:
    def __init__(self, path, datetime_start, datetime_end):
        self.path = path
        self.datetime_start = datetime_start
        self.datetime_end = datetime_end
        self._fetch_data()

    def _fetch_data(self):
        self.df = pd.read_sql(self.path)


class SignalLog(Log):
    def __init__(self, path, datetime_start, datetime_end):
        super().__init__(path, datetime_start, datetime_end)
        self.path = path
        self.datetime_start = datetime_start
        self.datetime_end = datetime_end

    def process_df_signal(self):
        return self.df


class MarketLog(Log):
    def __init__(self, path, datetime_start, datetime_end):
        super().__init__(path, datetime_start, datetime_end)
        self.path = path
        self.datetime_start = datetime_start
        self.datetime_end = datetime_end

    def process_df_market(self):
        return self.df


class TradeLog:
    def __init__(self, path, datetime_start, datetime_end):
        super().__init__(path, datetime_start, datetime_end)
        self.path = path
        self.datetime_start = datetime_start
        self.datetime_end = datetime_end


class SingleCombinedRecord:
    def __init__(self, signal_id, log_signal_all, log_market_all):
        """

        Args:
            raw_signal (pd.Series): e.g. {'datetime', 'direction': 'long', 'limit_1': 3707.0, 'stop': 3697.0, 'limit_2': 3710.0, 'num:': 1, 'duration_min': 3, 'cancel_second': 60}
            df_market (pd.DataFrame): MarketLog
        """
        self.signal_id = signal_id
        self.log_signal_all = log_signal_all
        self.log_market_all = log_market_all
        self.raw_signal = log_signal_all.df[signal_id]
        self._get_interval_market()
        self._check_signal()

    def _get_interval_market(self):
        """获取信号作用区间的行情信息

        Returns:
            pd.DataFrame: 信号作用区间的行情信息
        """
        datetime_begin = self.raw_signal["datetime"]
        datetime_end = datetime_begin + datetime.timedelta(
            minutes=self.raw_signal["duration_min"]
        )
        self.market_df_all = self.log_market_all.process_df_market()
        self.market_df_interval = self.market_df_all.loc[datetime_begin:datetime_end, :]

    def _check_signal(self):
        """判断信号的对错

        Returns:
            pd.DataFrame: 信号对错判断True/False
        """
        self.interval_max = self.market_df_interval["last_price"].max()
        self.interval_min = self.market_df_interval["last_price"].min()
        if self.raw_signal["direction"] == "long":
            if self.raw_signal["limit_2"] <= self.interval_max:
                self.correct = True
            else:
                self.correct = False
        elif self.raw_signal["direction"] == "short":
            if self.raw_signal["limit_2"] >= self.interval_min:
                self.correct = True
            else:
                self.correct = False


class AnalysisLog:
    """需要分析的内容包括（使用joblib，同时分析多个信号）：

    1. 信号个数统计
    2. 正确信号个数统计
    3. 市场行情中，所有满足条件的时点（3.1 实验所有tick；3.2 实验路径依赖性）
    4.

    """

    def __init__(
        self,
        log_market_all: MarketLog,
        log_signal_all: SignalLog,
        log_trade_all: TradeLog,
    ):
        self.log_market_all = log_market_all
        self.log_signal_all = log_signal_all
        self.log_trade_all = log_trade_all

    def count_signal(self):
        self.num_signal = self.log_signal_all.df.shape[0]
        return self.num_signal

    def count_signal_right(self, n_job):
        signal_right_lst = Parallel(n_jobs=n_job)(
            delayed(self.count_signal_utils)(
                signal_id, self.log_signal_all, self.log_market_all
            )
            for signal_id in self.log_signal_all.index.values()
        )
        self.num_signal_right = sum(signal_right_lst)
        return self.num_signal_right

    @staticmethod
    def count_signal_utils(signal_id, signal_all, market_all):
        """判断单个信号的正误"""
        single_combined_log = SingleCombinedRecord(signal_id, signal_all, market_all)
        return single_combined_log.correct
