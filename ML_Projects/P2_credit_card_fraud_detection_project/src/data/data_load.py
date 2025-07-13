# مربوط به گام اول
# src/data/data_load.py
import pandas as pd

def data_load(file_path):
    """
    بارگذاری داده‌ها از یک مسیر کامل فایل CSV.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"✅ داده‌ها از مسیر {file_path} با موفقیت بارگذاری شدند.")
        return data
    except FileNotFoundError:
        print(f"❌ خطا: فایل در مسیر {file_path} پیدا نشد!")
        return None
