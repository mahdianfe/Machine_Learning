# مربوط به گام دوم
# src/data/validate_data.py

import pandas as pd

def data_validate(df: pd.DataFrame):
    """
    بررسی کیفیت داده‌های دیتافریم df
    اگر داده مشکل داشته باشد، Exception ایجاد می‌شود.
    """

    # بررسی وجود مقادیر خالی
    # # .any() اول مشخص می‌کند کدام ستون‌ها NaN دارند، و .any() دوم بررسی می‌کند که آیا حداقل یک ستون NaN دارد یا نه.
    if df.isnull().any().any():
        raise ValueError("داده دارای مقادیر خالی (NaN) است.")

    # بررسی اینکه ستون Class که در creditcard.csv است فقط شامل 0 و 1 باشد
    if not df['Class'].isin([0, 1]).all():
        raise ValueError("ستون Class شامل مقادیر غیرمجاز است.")

    # بررسی اینکه ستون Amount منفی نباشد
    if (df['Amount'] < 0).any():
        raise ValueError("مقدار منفی در ستون Amount یافت شد.")

    # بررسی نامتعارف بودن مقادیر در ستون Time (مثلاً خیلی بزرگ یا کوچک)
    if df['Time'].max() > 172800:  # مثلاً اگر زمان بیش از 2 روز باشد
        print("⚠️ هشدار: مقادیر زمان بالاتر از 48 ساعت یافت شد.")

    print("✅ اعتبارسنجی داده با موفقیت انجام شد.")
