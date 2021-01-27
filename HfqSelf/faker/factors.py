import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class OvernightJumpFactor(BaseEstimator, TransformerMixin):
    # 隔夜价差因子
    def __init__(self, n_period=1):
        self.n_period = n_period

    def transform(self, cubic_data) -> pd.DataFrame:
        c = cubic_data.select_field("close")
        o = cubic_data.select_field("open")

        c_previous = c.shift()
        jump = o / c_previous - 1
        result = jump.rolling(self.n_period).mean()
        return result