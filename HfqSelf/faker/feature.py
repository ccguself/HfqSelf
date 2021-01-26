import os
import pytorch
from numpy as np
from pandas as pd
from .data import RawData

class FactorData:
    def __init__(self, raw_data:RawData):
        self.raw_data = raw_data
    
    
    
