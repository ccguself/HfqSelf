import os
import pytorch
import numpy as np
import pandas as pd
from data import RawData
from joblib import Parallel, delayed
from config import data_config
import datetime

# 完成因子筛选（纵），样本集的生成（横）


class FactorData:
    def __init__(self, raw_data: RawData):
        self.raw_data = raw_data

    # resample时，需要对10：15，11：30这些数据，做特殊处理：在resample之前，截取时候需要跨过休市时间；resample过后，要dropna
    def resample_and_load_factor_zoo_artificial(self, n_job):
        def resample_and_load_factor_zoo_artificial_utils(
            X, timestamp, config=data_config
        ):
            # step1. 向“过去”截取用于训练的时间区间
            interval_begin = timestamp - datetime.timedelta(
                seconds=config["window_resample_second"]
            )
            interval_end = timestamp

            X["datetime"]
            return

        data_slice_factors_added_lst = Parallel(n_jobs=n_job)(
            delayed(resample_and_load_factor_zoo_artificial_utils)(datetime_slice)
            for datetime_slice in datetime_sampled_series_lst
        )

        return
