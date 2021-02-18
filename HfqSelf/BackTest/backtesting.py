import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


class ModelInfo:
    def __init__(self, name_model, window_train, window_predict):
        self.name_model = name_model
        self.window_train = window_train
        self.window_predict = window_predict

    def _load_model(self):
        # 加载模型
        # self.model =
        return


class DailyLog:
    def __init__(self, date, market_data, signal_lst, signal_duration):
        self.date = date
        self.market_data = market_data
        self.signal_lst = signal_lst
        self._cal_profit()
        self._cal_acc()
        self._cal_reacall()

    def _cal_profit(self):
        def _cal_profit_utils(signal, mode="active"):
            """计算单个信号的收益情况

            Args:
                signal (dict): {"direction":"long", "datetime":datetime.datime}
                mode (str, optional): active or passive(主买主卖/被动挂单). Defaults to "active".
            """

            # step1. 首先截取信号有效区间的行情

            # step2. 根据不同的模式，对信号的收益进行判断

            # step3.
            return

        return

    # def _cal_acc(self):
    #     def _cal_acc_utils():
    #         return

    #     return

    # def _cal_reacall(self):
    #     def _cal_recall_utils():
    #         return

    #     return


class MarketData:
    def __init__(self, path_data, date_begin, date_end):
        self.path_data = path_data
        self.date_begin = date_begin
        self.date_end = date_end
        self.file_paths = []

    def _get_trading_dates(self):
        names = list(
            filter(
                lambda x: x >= self.date_begin and x <= self.date_end,
                os.listdir(self.path_data),
            )
        )
        self.file_paths.extend(
            [os.path.join(self.path_data, name) for name in names])
        self.file_paths.sort()

    # 只获取信号的时间戳
    def generate_signal(self, n_job=4):
        def _generate_signal_utils(
            daily_file_path: str, model_info: ModelInfo
        ) -> DailyLog:
            daily_date = daily_file_path.split(".")[0][-8:]  # 截取日期
            daily_market_data = pd.read_csv(daily_file_path)

            # 获取 需要预测的时点

            # 获取 开仓信号

            return

    # 根据信号的时间戳，计算净值
