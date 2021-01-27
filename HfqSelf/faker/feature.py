import os
import pytorch
from numpy as np
from pandas as pd
from .data import RawData

# 完成因子筛选（纵），样本集的生成（横）

class FactorData:
    def __init__(self, raw_data:RawData):
        self.raw_data = raw_data
    
    
    
