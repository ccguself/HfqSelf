default_config = {
    "used_zl": ["ZL00", "ZL01"],
    "mode_merge_zl": "cross_section",
    "used_time": ["night", "morning_1", "morning_2", "noon"],
    "size_resample": 15,  # resample的 “时间/s” 颗粒度
    "size_train": 75,  # resample后 “K线/个” 的颗粒度计算(通过resample，得到的是时间颗粒度)
    "size_predict": 3,  # resample前 “时间/min” 的颗粒度计算（通过resample，得到的是时间颗粒度）
    "size_step_increment": 1,  # 划分样本时的间隔，“K线/个” 的颗粒度计算
    "slice_overlap": False,
}