# در این فایل از کراس‌ولیدیشن برای ارزیابی استفاده می‌کنیم.

#کراس ولیدیشن باید از داده ای که قبلا به تست و آموزشی تقسیم شده انتخاب شود
# 1.برای جلوگیری از نشت اطلاعات (Data Leakage)
# به این ترتیب، مجموعه آزمایشی واقعاً برای ارزیابی نهایی و غیرجانبدارانه مدل انتخاب‌شده استفاده می‌شود.
    # 2.تسریع فرآیند آموزش و ارزیابی در مرحله انتخاب مدل: کراس‌ولیدیشن می‌تواند از نظر محاسباتی گران باشد،
# به خصوص اگر مجموعه داده بزرگی داشته باشید یا از مدل‌های پیچیده‌ای استفاده کنید.
    # ارزیابی عملکرد کلی مدل‌ها قبل از تنظیم دقیق: در مرحله train_models، هدف ممکن است ارزیابی سریع عملکرد کلی چند مدل مختلف
# با استفاده از کراس‌ولیدیشن بر روی داده‌های آموزشی باشد
# تا مدل‌های امیدوارکننده‌تر برای تنظیم دقیق‌تر در مراحل بعدی انتخاب شوند.
#____________________________________________
# # مربوط به گام هفتم
# src/model_training.py

import joblib
import pandas as pd # این ایمپورت لازمه، چون X_train و y_train از نوع DataFrame/Series هستند
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline # این هم لازمه اگر از Pipeline در نوع‌دهی (type hints) استفاده کنید یا بخواهید مدل‌های بیشتری اضافه کنید.

# ایمپورت پایپ‌لاین‌های تعریف شده از src/pipeline.py
from src.pipeline import (
    build_logistic_regression_pipeline,
    build_random_forest_pipeline,
    build_svm_pipeline
)
# اگر فقط build_pipeline را نگه داشته‌اید:
# from src.pipeline import build_pipeline

def train_models(X_train: pd.DataFrame, y_train: pd.Series):
    """
    آموزش مدل‌های یادگیری ماشین.

    Args:
        X_train (pd.DataFrame): مجموعه داده آموزش (ویژگی‌ها).
        y_train (pd.Series): مجموعه داده آموزش (متغیر هدف).

    Returns:
        dict: دیکشنری شامل مدل‌های آموزش دیده.
    """
    models = {
        "Logistic Regression": build_logistic_regression_pipeline(),
        "Random Forest": build_random_forest_pipeline(),
        "SVM": build_svm_pipeline()
    }

    # در اینجا، X_train و y_train به صورت مقیاس‌بندی نشده وارد می‌شوند.
    # هر پایپ‌لاین خودش StandardScaler را در مرحله fit اعمال می‌کند.
    for name, model in models.items():
        model.fit(X_train, y_train)
        # می‌توانید اینجا گزارش طبقه‌بندی را برای هر مدل چاپ کنید (روی داده آموزشی)
        y_pred_train = model.predict(X_train)
        print(f"\n--- Classification Report for {name} (Training Data) ---")
        print(classification_report(y_train, y_pred_train))

    return models
