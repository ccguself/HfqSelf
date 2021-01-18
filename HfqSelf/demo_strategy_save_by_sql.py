import datetime
import numpy as np
import pandas as pd
from dataclasses import dataclass
import torch
import datetime
import time
from vnpy.app.cta_strategy import (
    CtaTemplate,
    StopOrder,
    TickData,
    BarData,
    TradeData,
    OrderData,
    BarGenerator,
)
from vnpy.trader.constant import Status, Direction, Offset
import pandas as pd
import numpy as np

from vnpy.trader.code.predict import signal_output
from collections import deque
from peewee import *

db = SqliteDatabase(r"D:\Trader\.vntrader\database.db")


class SignalDb(Model):
    signal_id = CharField(primary_key=True)
    strategy = CharField()  # 产生该信号的策略名称
    symbol = CharField()  # 标的名称（rb2105.SHFE）

    datetime_begin = DateTimeField()  # 信号作用区间-开始
    datetime_begin = DateTimeField()  # 信号作用区间-结束
    direction = CharField()  # 方向
    price_limit_1 = FloatField()  # 开仓价格
    price_limit_2 = FloatField()  # 平仓价格
    price_stop = FloatField()  # 强制止损价格

    volume_signal = IntegerField()  # 信号下单量
    duration_min = IntegerField()  # 持仓时间/分钟
    cancel_second = IntegerField()  # 撤单时间/秒

    used_to_trade = BooleanField()  # 是否触发交易

    class Meta:
        database = db


class TradeSimulateDb(Model):
    strategy = CharField()  # 产生该信号的策略名称
    symbol = CharField()  # 标的名称（rb2105.SHFE）

    datetime_simulate_trade = DateTimeField()  # 模拟盘中，发生该笔交易的时间
    datetime_triger_tick = DateTimeField()  # 模拟盘中，触发该交易的tick的时间

    direction = CharField()  # 多空方向
    offset = CharField()  # 开平仓方向
    #     price_trade_ideal = FloatField()  # 理想成交价格
    price_trade_simulate = FloatField()  # 模拟盘成交价格

    #     volume_trade_ideal = IntegerField()  # 策略理想下单量
    volume_trade_simulate = IntegerField()  # 策略模拟盘下单量

    class Meta:
        database = db
        primary_key = CompositeKey("datetime_triger_tick", "direction")


class TickCollector:
    def __init__(
        self,
        window_collect: int = 2000,
        window_train: int = 2000,
        frequency_predict: int = 20,
    ):
        self.window_collect = window_collect
        self.window_train = window_train
        self.frequency_predict = frequency_predict
        self.count: int = 0
        self.inited: bool = False
        self.prophet = signal_output()

        self.datetime_array: np.ndarray = np.zeros(window_collect)
        self.LaxtPx_array: np.ndarray = np.zeros(window_collect)
        self.Buy1Price_array: np.ndarray = np.zeros(window_collect)
        self.Buy1OrderQty_array: np.ndarray = np.zeros(window_collect)
        self.Sell1Price_array: np.ndarray = np.zeros(window_collect)
        self.Sell1OrderQty_array: np.ndarray = np.zeros(window_collect)

        self.TotalVolumeTrade_array: np.ndarray = np.zeros(window_collect)
        self.OpenInterest_array: np.ndarray = np.zeros(window_collect)

    def update_tick(self, tick) -> None:
        self.count += 1
        if not self.inited and self.count >= self.window_train:
            self.inited = True

        self.datetime_array[:-1] = self.datetime_array[1:]
        self.Buy1Price_array[:-1] = self.Buy1Price_array[1:]
        self.Buy1OrderQty_array[:-1] = self.Buy1OrderQty_array[1:]
        self.Sell1Price_array[:-1] = self.Sell1Price_array[1:]
        self.Sell1OrderQty_array[:-1] = self.Sell1OrderQty_array[1:]
        self.LaxtPx_array[:-1] = self.LaxtPx_array[1:]
        self.TotalVolumeTrade_array[:-1] = self.TotalVolumeTrade_array[1:]
        self.OpenInterest_array[:-1] = self.OpenInterest_array[1:]

        self.datetime_array[-1] = tick.datetime.timestamp()
        self.Buy1Price_array[-1] = tick.bid_price_1
        self.Buy1OrderQty_array[-1] = tick.bid_volume_1
        self.Sell1Price_array[-1] = tick.ask_price_1
        self.Sell1OrderQty_array[-1] = tick.ask_volume_1
        self.LaxtPx_array[-1] = tick.last_price
        self.TotalVolumeTrade_array[-1] = tick.volume
        self.OpenInterest_array[-1] = tick.open_interest

        self.df_info = np.c_[
            self.datetime_array,
            self.LaxtPx_array,
            self.TotalVolumeTrade_array,
            self.Buy1Price_array,
            self.Buy1OrderQty_array,
            self.Sell1Price_array,
            self.Sell1OrderQty_array,
        ]

    def generate_signal(self):
        signal_lst = self.prophet.get_output_list(self.df_info)
        return signal_lst


class SignalData:
    def __init__(self, raw_signal, timestamp, name_strategy, name_label, tick_price):
        self.tick_price = tick_price
        self.direction = raw_signal["direction"]
        self.price_opening = raw_signal["limit_1"]
        self.price_closing = raw_signal["limit_2"]
        self.price_stop_loss = raw_signal["stop"]
        self.volume = raw_signal["num"]
        self.duration_min = datetime.timedelta(minutes=int(raw_signal["duration_min"]))
        self.cancel_second = datetime.timedelta(
            seconds=int(raw_signal["cancel_second"])
        )

        self.datetime = timestamp
        self.time_cancel = self.datetime + self.cancel_second
        self.time_invalid = self.datetime + self.duration_min

        self.name_strategy = name_strategy
        self.name_label = name_label
        self.signal_id = (
            self.name_strategy
            + "_"
            + self.name_label
            + "_"
            + self.direction
            + "_"
            + str(raw_signal["duration_min"])
            + "_"
            + self.datetime.strftime("%Y-%m-%d %H:%M:%S")
        )


class BangStrategy(CtaTemplate):

    author = "ChiZhang"

    frequency_forecast = 20  # tick个数
    number_signal_to_trade = 3

    parameters = ["frequency_forecast" "number_signal_to_trade"]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        """"""
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

        self.tc = TickCollector(frequency_predict=self.frequency_forecast)

        # 基于信号的时间状态更新
        self.signal_lst = deque(maxlen=self.number_signal_to_trade)
        self.signal_count_long = 0
        self.signal_count_short = 0

    def on_init(self):
        """
        Callback when strategy is inited.
        """
        self.write_log("策略初始化")
        self.load_tick(1)

    def on_start(self):
        """
        Callback when strategy is started.
        """
        self.write_log("策略启动")

    def on_stop(self):
        """
        Callback when strategy is stopped.
        """
        self.write_log("策略停止")

    def on_tick(self, tick: TickData):
        """
        Callback of new tick data update.
        """
        self.tc.update_tick(tick)
        if self.tc.inited == False:
            return
        tick_time = tick.datetime.replace(tzinfo=None)
        raw_signals = self.tc.generate_signal()

        for i, raw_signal in enumerate(raw_signals):
            signal_data = SignalData(
                raw_signal,
                tick_time,
                self.strategy_name,
                self.vt_symbol,
                tick.last_price,
            )
            # 信号落库
            try:
                SignalDb.create(
                    strategy=signal_data.name_strategy,
                    symbol=signal_data.name_label,
                    signal_id=signal_data.signal_id,
                    datetime_begin=signal_data.datetime,
                    datetime_end=signal_data.time_invalid,
                    direction=signal_data.direction,
                    price_limit_1=signal_data.price_opening,
                    price_limit_2=signal_data.price_closing,
                    price_stop=signal_data.price_stop_loss,
                    volume_signal=signal_data.volume,
                    duration_min=raw_signal["duration_min"],
                    cancel_second=raw_signal["cancel_second"],
                    used_to_trade=False,
                )
            except:
                return

        # 发出订单，落库放在on_trade中
        if self.tc.count % self.tc.frequency_predict == 0:
            long_pos = self.cta_engine.offset_converter.get_position_holding(
                self.vt_symbol
            ).long_pos
            short_pos = self.cta_engine.offset_converter.get_position_holding(
                self.vt_symbol
            ).short_pos
            self.signal_lst.append(raw_signals)
            if len(self.signal_lst) < self.number_signal_to_trade:
                return
            self.signal_count_long = 0
            self.signal_count_short = 0
            for i, signals in enumerate(self.signal_lst):
                long_signal = [x for x in signals if x["direction"] == "long"]
                short_signal = [x for x in signals if x["direction"] == "short"]
                self.signal_count_long += len(long_signal)
                self.signal_count_short += len(short_signal)

            long_signal_now = [x for x in raw_signals if x["direction"] == "long"]
            short_signal_now = [x for x in raw_signals if x["direction"] == "short"]

            if long_pos == 0:
                # raw_signals中提取做多信号
                if self.signal_count_long == self.number_signal_to_trade:

                    raw_signal = long_signal_now[0]
                    signal_data = SignalData(
                        raw_signal,
                        tick_time,
                        self.strategy_name,
                        self.vt_symbol,
                        tick.last_price,
                    )
                    vt_orderid_lst = self.buy(
                        signal_data.price_opening + 10, signal_data.volume, False
                    )
                    q = SignalDb.update({SignalDb.used_to_trade: True}).where(
                        SignalDb.signal_id == signal_data.signal_id
                    )
                    q.execute()
                    try:
                        TradeSimulateDb.create(
                            strategy=self.strategy_name,
                            symbol=self.vt_symbol,
                            datetime_simulate_trade=tick.datetime,
                            datetime_triger_tick=tick.datetime,
                            direction="long",
                            offset="open",
                            price_trade_simulate=tick.last_price,
                            volume_trade_simulate=tick.volume,
                        )
                    except:
                        return

            else:
                if self.signal_count_long == 0:
                    vt_orderid_lst = self.sell(tick.last_price - 10, long_pos, False)
                    try:
                        TradeSimulateDb.create(
                            strategy=self.strategy_name,
                            symbol=self.vt_symbol,
                            datetime_simulate_trade=tick.datetime,
                            datetime_triger_tick=tick.datetime,
                            direction="short",
                            offset="close",
                            price_trade_simulate=tick.last_price,
                            volume_trade_simulate=tick.volume,
                        )
                    except:
                        return

            if short_pos == 0:
                # raw_signals中提取做多信号
                if self.signal_count_short == self.number_signal_to_trade:

                    raw_signal = short_signal_now[0]
                    signal_data = SignalData(
                        raw_signal,
                        tick_time,
                        self.strategy_name,
                        self.vt_symbol,
                        tick.last_price,
                    )
                    vt_orderid_lst = self.short(
                        signal_data.price_opening - 10, signal_data.volume, False
                    )
                    q = SignalDb.update({SignalDb.used_to_trade: True}).where(
                        SignalDb.signal_id == signal_data.signal_id
                    )
                    q.execute()
                    try:
                        TradeSimulateDb.create(
                            strategy=self.strategy_name,
                            symbol=self.vt_symbol,
                            datetime_simulate_trade=tick.datetime,
                            datetime_triger_tick=tick.datetime,
                            direction="short",
                            offset="open",
                            price_trade_simulate=tick.last_price,
                            volume_trade_simulate=tick.volume,
                        )
                    except:
                        return

            else:
                if self.signal_count_short == 0:
                    vt_orderid_lst = self.cover(tick.last_price + 10, short_pos, False)
                    try:
                        TradeSimulateDb.create(
                            strategy=self.strategy_name,
                            symbol=self.vt_symbol,
                            datetime_simulate_trade=tick.datetime,
                            datetime_triger_tick=tick.datetime,
                            direction="long",
                            offset="close",
                            price_trade_simulate=tick.last_price,
                            volume_trade_simulate=tick.volume,
                        )
                    except:
                        return

    def on_bar(self, bar: BarData):
        pass

    def on_trade(self, trade: TradeData):

        # 更新TradeSimulateDb
        pass

    def on_order(self, order: OrderData):
        pass

    def on_stop_order(self, stop_order: StopOrder):
        """
        Callback of stop order update.
        """
        pass
