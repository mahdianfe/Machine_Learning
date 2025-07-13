# مربوط به گام چهارم

# src/feature_engineering.py

from src.data.data_preprocess import data_preprocess
import pandas as pd
from src.data.data_preprocess import data_preprocess

def feature_engineering(data):
    data = data_preprocess(data)
    X = data.drop('Class', axis=1) # X should be a DataFrame with original column names
    y = data['Class']
    return X, y # Return X as a DataFrame