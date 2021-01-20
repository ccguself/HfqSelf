import os
import numpy as np
import pandas as pd
import seaborn as sns
import datetime
from joblib import Parallel, delayed
from peewee import *
from prophet.predict import signal_output


class SingleCombinedLog:
    def __init__(
        self,
        raw_signal,
        df_predict_interval,
        strategy_name="NN",
        direction="Long",
        duration_minute=2,
        pct_predict=2,
    ):
        self.strategy_name = strategy_name
        self.direction = direction
        self.duration_minute = duration_minute
        self.pct_predict = pct_predict
        self.raw_signal = raw_signal
        self.df_predict_interval = df_predict_interval
        self._check_signal()

    def _check_signal(self):
        interval_begin = self.df_predict_interval["last_price"].iloc[0]
        interval_max = self.df_predict_interval["last_price"].max()
        interval_min = self.df_predict_interval["last_price"].min()
        if self.direction == "Long":
            if (interval_begin + self.pct_predict) < interval_max:
                if self.raw_signal == None:
                    self.correct = "FT"
                else:
                    self.correct = "TT"
            else:
                if self.raw_signal == None:
                    self.correct = "FF"
                else:
                    self.correct = "TF"
        elif self.direction == "Short":
            if (interval_begin - self.pct_predict) > interval_min:
                if self.raw_signal == None:
                    self.correct = "FT"
                else:
                    self.correct = "TT"
            else:
                if self.raw_signal == None:
                    self.correct = "FF"
                else:
                    self.correct = "TF"

        return self.correct


class MarketLog:
    def __init__(
        self,
        db_path="D:\Trader\.vntrader\database.db",
        datetime_start="2021-01-17",
        datetime_end="2021-01-18",
        symbol="rb2105",
    ):
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
        # self.df_market["datetime"] = self.df_market["datetime"].apply(
        #     lambda x: x.timestamp()
        # )
        return self.df_market


class AnalysisSignal:
    def __init__(self, analysised_signal_dict):
        self.analysised_signal_dict = analysised_signal_dict

    def count_signal(self):
        signal_lst = list(self.analysised_signal_dict.values())
        valid_signal_lst = [x for x in signal_lst if x.raw_signal != None]
        self.num_signal_total = len(valid_signal_lst)
        return self.num_signal_total

    def cal_acc(self):
        signal_lst = list(self.analysised_signal_dict.values())
        num_tf = len([x for x in signal_lst if x.correct == "TF"])
        num_tt = len([x for x in signal_lst if x.correct == "TT"])
        num_ft = len([x for x in signal_lst if x.correct == "FT"])
        num_ff = len([x for x in signal_lst if x.correct == "FF"])
        self.acc = num_tt / (num_tf + num_tt)
        return self.acc

    def cal_recall(self):
        signal_lst = list(self.analysised_signal_dict.values())
        num_tf = len([x for x in signal_lst if x.correct == "TF"])
        num_tt = len([x for x in signal_lst if x.correct == "TT"])
        num_ft = len([x for x in signal_lst if x.correct == "FT"])
        num_ff = len([x for x in signal_lst if x.correct == "FF"])
        self.recall = num_tt / (num_tt + num_ft)
        return self.recall

    def generate_order(self):
        pass


class AnalysisLog:
    """一些注意点：
    1. dict_...会分别记录各个类型的信号，不同模型发出的，不同有效时段的，不同方向的
    """

    def __init__(
        self, total_market_log: MarketLog, pre_start_interval=2000, frequency_predict=20
    ):
        self.total_market_log = total_market_log
        self.pre_start_interval = pre_start_interval  # 模型预启动需要使用的tick数量
        self.frequency_predict = frequency_predict
        self.dict_nn_long_two = {}  # datetime: SingleCombinedLog
        self.dict_nn_short_two = {}  # datetime: SingleCombinedLog
        # self.result_analysis =
        self.model = signal_output()

    def generate_single_combined_log(self):
        # 逐tick数据滑动，生成信号
        df_market_total = self.total_market_log.df_market
        for i, timestamp_beging in enumerate(df_market_total["datetime"]):
            # 判断多空信号
            if (i > self.pre_start_interval) & ((i % self.frequency_predict) == 0):
                train_df = df_market_total.iloc[i - self.pre_start_interval : i, :]
                train_df["datetime"] = train_df["datetime"].apply(
                    lambda x: x.timestamp()
                )
                raw_signals = self.model.get_output_list(train_df.values)

                # 拆分信号
                lst_signal_nn_long_two = [
                    x
                    for x in raw_signals
                    if ((x["direction"] == "long") & (x["duration_min"] == 2))
                ]
                lst_signal_nn_short_two = [
                    x
                    for x in raw_signals
                    if ((x["direction"] == "short") & (x["duration_min"] == 2))
                ]

                # 更新dict
                timestamp_end = timestamp_beging + datetime.timedelta(minutes=2)
                df_predict = df_market_total.loc[
                    (df_market_total["datetime"] > timestamp_beging)
                    & (df_market_total["datetime"] < timestamp_end)
                ]

                ## 2分钟的信号
                # lst_signal_nn_long_two
                if len(lst_signal_nn_long_two) == 0:
                    self.dict_nn_long_two[timestamp_beging] = SingleCombinedLog(
                        strategy_name="NN",
                        direction="Long",
                        duration_minute=2,
                        pct_predict=2,
                        raw_signal=None,
                        df_predict_interval=df_predict,
                    )
                else:
                    signal = lst_signal_nn_long_two[0]
                    self.dict_nn_long_two[timestamp_beging] = SingleCombinedLog(
                        strategy_name="NN",
                        direction="Long",
                        duration_minute=2,
                        pct_predict=2,
                        raw_signal=signal,
                        df_predict_interval=df_predict,
                    )
                # lst_signal_nn_short_two
                if len(lst_signal_nn_short_two) == 0:
                    self.dict_nn_short_two[timestamp_beging] = SingleCombinedLog(
                        strategy_name="NN",
                        direction="Short",
                        duration_minute=2,
                        pct_predict=2,
                        raw_signal=None,
                        df_predict_interval=df_predict,
                    )
                else:
                    signal = lst_signal_nn_short_two[0]
                    self.dict_nn_short_two[timestamp_beging] = SingleCombinedLog(
                        strategy_name="NN",
                        direction="Short",
                        duration_minute=2,
                        pct_predict=2,
                        raw_signal=signal,
                        df_predict_interval=df_predict,
                    )
