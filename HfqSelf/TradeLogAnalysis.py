import os
import numpy as np
import pandas as pd
import seaborn as sns
import datetime


class SingleLog:
    def __init__(self, raw_signal, data_market):
        """

        Args:
            raw_signal (pd.Series): e.g. {'datetime', 'direction': 'long', 'limit_1': 3707.0, 'stop': 3697.0, 'limit_2': 3710.0, 'num:': 1, 'duration_min': 3, 'cancel_second': 60}
            df_market (pd.DataFrame): 全部（当天）的行情信息
        """
        self.raw_signal = raw_signal
        self.data_market = data_market
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
        self.interval_market = self.data_market.loc[datetime_begin:datetime_end, :]
        return self.interval_market

    def _check_signal(self):
        """判断信号的对错

        Returns:
            pd.DataFrame: 信号对错判断True/False
        """

        return


class SignalLog:
    def __init__(self, path, datetime_start, datetime_end):
        self.path = path
        self.datetime_start = datetime_start
        self.datetime_end = datetime_end

    def fetch_data_signal(self):
        self.df_signal = pd.read_sql(self.path)
        self.df_signal_processed = self.process_df_signal(self.df_signal)
        return self.df_signal

    @staticmethod
    def process_df_signal(df):
        return df


class MarketLog:
    def __init__(self, path, datetime_start, datetime_end):
        self.path = path
        self.datetime_start = datetime_start
        self.datetime_end = datetime_end

    def fetch_data_market(self):
        self.df_market = pd.read_sql(self.path)
        self.df_market_processed = self.process_df_market(self.df_market)
        return self.df_market_processed

    @staticmethod
    def process_df_market(df):
        return df


class AnalysisLog:
    def __init__(self, data_market: MarketLog, date_signal: SignalLog):
        self.data_market = data_market
        self.data_signal = date_signal
