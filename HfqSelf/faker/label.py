from config import data_config
import datetime
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import Parallel, delayed


class GetLabel(BaseEstimator, TransformerMixin):
    def __init__(self, data_config, n_job):
        self.data_config = data_config
        self.n_job = n_job

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        def transform_utils(time_index, signal_duration_min, df):
            label_dict = {}
            interval_min = datetime.timedelta(minutes=signal_duration_min)
            df_interval = df.loc[time_index:time_index + interval_min, :]

            # label_1 : pct of Mid Price
            label_dict["label_midprice_pct"] = df_interval["mid_price"].iloc[-1] / \
                df_interval["mid_price"].iloc[0]

            # label_2 : …………

            # label_3 : …………

            # label_4 : …………

            # label_5 : …………

            return label_dict

        label_dict_lst = Parallel(n_jobs=self.n_job)(
            delayed(transform_utils)(
                time_index, self.data_config["window_predict_minute"], X)
            for time_index in list(X.index)
        )
        key_name_lst = list(label_dict_lst[0].keys())
        for key_name in key_name_lst:
            X[key_name] = [x[key_name] for x in label_dict_lst]
        return X
