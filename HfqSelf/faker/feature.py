import os
import pytorch
import numpy as np
import pandas as pd
from .data import RawData

# 完成因子筛选（纵），样本集的生成（横）


class FactorData:
    def __init__(self, raw_data: RawData):
        self.raw_data = raw_data

    # resample时，需要对10：15，11：30这些数据，做特殊处理：在resample之前，截取时候需要跨过休市时间；resample过后，要dropna
    def load_factor_zoo_artificial(self, n_job):
        def load_factor_zoo_artificial_utils():
            # step1.
            return

        return
