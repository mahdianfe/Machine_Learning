# main.py

import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# اضافه کردن مسیر پروژه برای import صحیح
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
# مهم : روی import ماژول‌ها اثر داره.
# و بدون این خط  پایتون نمیفهمه مثلا فولدر اس ار سی کجاست و ایمپورت ها ارورو میدن
# توجه که os.path.join()فقط دو تا چیز را کنار هم قرار میده و تفسیر نمیکنه
#و چون os.path.abspath دو تا ارگومان قبول نمیکنه از os.path.join استفاده کردیم که دو تا ادرس را فقط کنار هم قرار بده
# بعد os.path.abspath بیاد تفسیر کنه که '..' که داخل این مسیر بود یعنی یک دایرکتوری بیا عقب

from src.data.data_load import data_load
from src.data.data_validate import data_validate
from src.data.data_preprocess import data_preprocess
from src.feature_engineering import feature_engineering
from src.model_training import train_models
from src.model_evaluation import evaluate_models_cv, plot_roc_curves
from src.hyperparameter_tuning import tune_hyperparameters
from src.anomaly_detection import detect_anomalies
from src.data.data_split import split_and_save_data  # ایمپورت تابع تقسیم داده
from sklearn.metrics import f1_score

def print_stage(stage_name):
    print(f"\n{'='*60}")
    print(f"🚀 [گام {stage_name}]")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    # BASE_DIR به ریشه اصلی پروژه بستگی دارد
    # یعنی اگر فایل اصلی اجرایی در فولدر دیگه بود برای اینکه بیس دیر به ریشه اصلی پروژه برگردد پارامتر ".." را هم قرار میدادیم
    input_path = os.path.join(BASE_DIR, 'data', 'raw', 'creditcard.csv')
    processed_dir = os.path.join(BASE_DIR, 'data', 'processed')
    # BASE_DIR را اوردیم چون مسیر نسبی رو قبول نمیکرد
    outputs_dir = os.path.join(BASE_DIR, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    # 🚀 گام 1: بارگذاری داده خام
    print_stage("1: بارگذاری داده خام")
    data = data_load(input_path)

    # 🚀 گام 2: اعتبارسنجی داده‌ها
    print_stage("2: اعتبارسنجی داده‌ها")
    data_validate(data)

    # 🚀 گام 3: پیش‌پردازش داده‌ها
    print_stage("3: پیش‌پردازش داده‌ها")
    data = data_preprocess(data)

    # 🚀 گام 4: مهندسی ویژگی‌ها
    print_stage("4: مهندسی ویژگی‌ها")
    X, y = feature_engineering(data)

    # 🚀 گام 5: تقسیم داده‌ها و ذخیره (تغییر یافته)
    print_stage("5: تقسیم داده‌ها و ذخیره")

    # X و y را مستقیماً اینجا تقسیم کنید
    X_train_df, X_test_df, y_train_series, y_test_series = train_test_split(
        X, y, test_size=0.2, random_state=42  # یا از test_size و random_state پیش‌فرض استفاده کنید
    )

    # مسیر کامل پوشه خروجی (processed_dir قبلاً تعریف شده است)
    os.makedirs(processed_dir, exist_ok=True)
    # این خط کد اطمینان حاصل می‌کند که:
    # پوشه data/processed قبل از اینکه بخواهید فایل‌های X_train.csv، X_test.csv و y_train.csv و y_test.csv را در آن ذخیره کنید، وجود داشته باشد.

    # ذخیره داده‌های تقسیم شده
    X_train_df.to_csv(os.path.join(processed_dir, "X_train.csv"), index=False)
    X_test_df.to_csv(os.path.join(processed_dir, "X_test.csv"), index=False)
    y_train_series.to_csv(os.path.join(processed_dir, "y_train.csv"), index=False)
    y_test_series.to_csv(os.path.join(processed_dir, "y_test.csv"), index=False)
    print(f"✅ داده‌های تقسیم شده در مسیر {processed_dir} ذخیره شدند.")


    # 🚀 گام 6: بارگذاری داده‌های تقسیم‌شده (تغییر یافته)
    print_stage("6: بارگذاری داده‌های تقسیم‌شده")
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))  # حالا X_train نام ستون‌های اصلی را دارد
    X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))  # حالا X_test نام ستون‌های اصلی را دارد

    # از .squeeze() استفاده میکنیم تا y یک pd.Series بماند:
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv")).squeeze()
    # متد .squeeze() در کتابخانه Pandas برای حذف ابعاد (dimensions) تک‌عنصری از یک DataFrame یا Series استفاده می‌شود.
    # یعن    ی اگر یک DataFrame فقط یک ستون یا یک ردیف داشته باشد،squeeze() آن را به یک Seriesتبدیل میکند
    # و اگر  Series فقط یک عنصر داشته باشد، .squeeze() آن را به یک اسکالر (یک مقدار تنها) تبدیل می‌کند.

    # 🚀 گام 7: آموزش مدل‌ها
    print_stage("7: آموزش مدل‌ها")
    models = train_models(X_train, y_train)


    # 🚀 گام 8: ارزیابی مدل‌ها با متریک‌های کامل
    print_stage("8: ارزیابی مدل‌ها")
    cv_results = evaluate_models_cv(models, X_train, y_train)

    # (اختیاری) نمایش نتایج نهایی
    print("\n📋 نتایج Cross-Validation:")
    for res in cv_results:
        print(f"✅ {res['Model']} - F1: {res['F1 Score']:.3f}, AUC: {res['ROC AUC']:.3f}")

    # 🚀 گام 9: رسم منحنی ROC
    print_stage("9: رسم منحنی ROC")
    plot_roc_curves(models, X_test, y_test)


    # 🚀 گام 10: تنظیم هایپرپارامترها
    print_stage("10: تنظیم هایپرپارامترها")
    tune_hyperparameters(X_train, y_train)

    # 🚀 گام 11: تشخیص و ذخیره ناهنجاری‌ها
    print_stage("11: تشخیص و ذخیره ناهنجاری‌ها")

    anomalies_path_base = os.path.join(outputs_dir, "anomalies")

    for model_name, model in models.items():
        anomalies_df = detect_anomalies(X_test, model=model, threshold=0.1)
        path = f"{anomalies_path_base}_{model_name.replace(' ', '_')}.csv"
        anomalies_df.to_csv(path, index=False)
        print(f"✅ ناهنجاری‌های مدل {model_name} در '{path}' ذخیره شدند.")


    # 🚀 گام 12: ذخیره مدل‌ها
    print_stage("12: ذخیره مدل‌ها")
    for model_name, model in models.items():
        joblib.dump(model, os.path.join(outputs_dir, f"{model_name}_pipeline.joblib"))
        print(f"✅ پایپ‌لاین {model_name} ذخیره شد.")


    # 🚀 گام 13: ذخیره نتایج ارزیابی در CSV
    print_stage("13: ذخیره نتایج در CSV")
    model_performance = {
        'Model': list(models.keys()),
        'Accuracy': [model.score(X_test, y_test) for model in models.values()], # ارزیابی روی داده تست
        'F1 Score': [0.0] * len(models) # نیاز به ارزیابی دقیق‌تر F1 Score
    }
    performance_df = pd.DataFrame(model_performance)
    performance_df.to_csv(os.path.join(outputs_dir, "model_performance_metrics.csv"), index=False)
    print("✅ نتایج در فایل CSV ذخیره شد.")


    print_stage("14: ذخیره نتایج در فایل متنی")

    with open(os.path.join(outputs_dir, "model_performance_metrics.txt"), "w", encoding="utf-8") as f:
        f.write("Model | Accuracy | F1\n")
        for name, acc, f1 in zip(model_performance['Model'],
                                 model_performance['Accuracy'],
                                 model_performance['F1 Score']):
            f.write(f"{name} | {acc:.4f} | {f1:.4f}\n")
    print("✅ نتایج در فایل متنی ذخیره شد.")


