import os
import numpy as np
import pandas as pd
import seaborn as sns
import datetime
from joblib import Parallel, delayed


class MarketLog:
    def __init__(self, path, datetime_start, datetime_end):
        self.path = path
        self.datetime_start = datetime_start
        self.datetime_end = datetime_end
        self._fetch_data()

    def _fetch_data(self):
        self.df = pd.read_sql(self.path)

    def generate_signals(self):
        pass
