import os
import pandas as pd
import random
from random import randint, shuffle
from datetime import datetime
import numpy as np
import time
from prophet.data import data_stct
from prophet.config import *
from prophet import forecast


class prediction:
    def __init__(self):
        self.model_rzbl = forecast.forecastNet(
            args_rzbl,
            save_file=r"D:\Code\hfq\HfqSelf\prophet\fcn_64d_s_6dim_3prd_rzbl_sig.pt",
        )
        self.model_cu = forecast.forecastNet(
            args_cu,
            save_file=r"D:\Code\hfq\HfqSelf\prophet\fcn_128d_s_5dim_3prd_cu.pt",
        )
        if args_rzbl.is_tick:
            self.model_rzbl = forecast.forecastNet(
                args_rzbl,
                save_file=r"D:\Code\hfq\HfqSelf\prophet\fcn_64d_s_9dim_3prd_05.pt",
            )
        self.sigmoid = lambda x: 1 / (1 + np.exp(-x / 5))
        self.dirct = ["long", "short"]
        self.weight = [1, 2]

    def get_datax(self, data_org, category="RZBL"):
        if category == "RZBL":
            args = args_rzbl
        if category == "CU":
            args = args_cu
        data_process = data_stct(args)
        data = data_process.pct(data_org)
        x = np.array(data[0])
        if len(x.shape) == 2:
            x = x[:, np.newaxis, :]
        else:
            x = np.transpose(x, (1, 0, 2))
        return x

    def run(self, data_org, x, d, tic, bound, intv=3, category="RZBL"):
        if category == "RZBL":
            args = args_rzbl
            fcstnet = self.model_rzbl
        if category == "CU":
            args = args_cu
            fcstnet = self.model_cu
        r = fcstnet.evaluate(x)

        def idx(a):
            return sum([args.out_seq_length // c for c in args.conv if c < a])

        # and r[idx(intv)][0][2]*(1-2*d)>0.5*(1-2*d):
        if r[idx(intv)][0][d] > self.sigmoid(tic) and r[0][0][2] * (1 - 2 * d) > 0.5 * (
            1 - 2 * d
        ):
            order_list = [
                {
                    "direction": self.dirct[d],
                    "limit_1": data_org[-1, 3] + i * (1 - 2 * d),
                    "stop": data_org[-1, 3] - 6 * (1 - 2 * d),
                    "limit_2": data_org[-1, 3] + bound * (1 - 2 * d),
                    "num": self.weight[i],
                    "duration_min": intv,
                    "cancel_second": 60,
                }
                for i in range(1)
            ]
            return order_list
        return []


class signal_output:
    def __init__(self):
        self.p = prediction()

    def get_output_list(self, data):
        x = self.p.get_datax(data)
        return self.p.run(data, x, 1, 2, 2, 2) + self.p.run(data, x, 0, 2, 2, 2)
