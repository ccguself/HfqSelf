from interval import Interval
import datetime

data_config = {
    "sample_mode": "tick",
    "window_train_minute": 10,
    "window_resample_second": 15,
    "window_predict_minute": 15,
    "freq_tick_per_second": 2
}


# def get_trading_and_break_interval_futures(symbol="RB", date="2020-01-18"):
#     if symbol == "RB":
#         date_today = datetime.strptime(date, "%Y-%m-%d").date()
#         date_yesterday = date_today - datetime.timedelta(days=1)

#         interval_trading_1 = Interval()
# interval_trading_lst =   # 用于盘间的休市时间判断
# interval_break =   # 用于夜盘和日盘间的判断
