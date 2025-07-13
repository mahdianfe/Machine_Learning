# مربوط به گام سوم
# src/data/data_preprocess.py

def data_preprocess(data):
    # پاکسازی داده‌ها: حذف مقادیر گمشده (اگر نیاز باشد)
    data = data.dropna()

    # در این نسخه فقط پاک‌سازی انجام می‌شه
    return data
