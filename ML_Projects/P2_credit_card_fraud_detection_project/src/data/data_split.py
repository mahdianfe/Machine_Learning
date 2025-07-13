# مربوط به گام پنجم -- غیر قابل استفاده -- در خود فایل مین انجام شده
# src/data/data_split.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split

def split_and_save_data(df: pd.DataFrame, target_column: str, output_dir: str = "data/processed",
                        test_size: float = 0.2, random_state: int = 42):
    """
    تقسیم داده به آموزش و تست و ذخیره آنها در فایل‌های CSV در مسیر مشخص.

    Args:
        df (pd.DataFrame): DataFrame حاوی داده‌ها.
        target_column (str): نام ستون هدف.
        output_dir (str): مسیر دایرکتوری برای ذخیره فایل‌های تقسیم شده (نسبت به ریشه پروژه).
        test_size (float, optional): نسبت اندازه مجموعه تست. پیش‌فرض 0.2.
        random_state (int, optional): seed برای تولید اعداد تصادفی. پیش‌فرض 42.
    """
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    # دوبار '..' را قرار دادیم زیرا اصل در BASE_DIR این است که در ریشه پروژه باشیم
    output_dir_full = os.path.join(BASE_DIR, output_dir)
    os.makedirs(output_dir_full, exist_ok=True)
    # آرگومان exist_ok=True یعنی اگر دایرکتوری (یا همان پوشه) که قصد ایجادش را دارید، از قبل وجود داشته باشد، خطایی رخ ندهد.

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # ذخیره داده‌ها
    X_train.to_csv(os.path.join(output_dir_full, "X_train.csv"), index=False)
    #آرگومان index=False مشخص می‌کند که آیا ستون ایندکس (شماره ردیف) DataFrame نیز باید در فایل CSV ذخیره شود یا خیر.
    X_test.to_csv(os.path.join(output_dir_full, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir_full, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir_full, "y_test.csv"), index=False)
    print(f"✅ داده‌های تقسیم شده در مسیر {output_dir_full} ذخیره شدند.")