# src/compare_models.py
# فایل ارزیابی ای است تا سه مدل Naive Bayes و  SVM و Logistic Regression را با هم مقایسه کند


import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#  رسم نمودار میله‌ای برای مقایسه مدل‌ها
def plot_model_comparison(results, save_path):
    labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    model_names = [r[0] for r in results]
    #این یک لیست‌ساز (list comprehension) است
    #برای هر عنصر r، اولین آیتم آن (با اندیس 0) انتخاب می‌شود.
    #در نهایت، یک لیست جدید به نام model_names ایجاد می‌شود که شامل نام تمام مدل‌های موجود در results است.

    scores = [r[1:] for r in results]
    # یک برش (slice) از آن ایجاد می‌شود که شامل تمام آیتم‌ها از اندیس 1 به بعد است.
    #در نهایت یک لیست جدید به نام scores ایجاد می‌شود که در آن هر عنصر، لیستی از نمرات ارزیابی برای یک مدل خاص است.
    scores = np.array(scores)
    # برای تبدیل لیست scores به یک آرایه NumPy استفاده می‌شود.
    # آرایه‌های NumPy برای انجام محاسبات عددی کارآمدتر هستند و برای رسم نمودار با matplotlib مناسب‌ترند.

    x = np.arange(len(labels))
    #با استفاده از np.arange یک آرایه NumPy از اعداد صحیح ایجاد می‌شود.

    width = 0.25 # این مقدار برای تعیین پهنای میله‌های نمودار
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(len(model_names)):
        ax.bar(x + i * width, scores[i], width, label=model_names[i])
        #x + i * width: موقعیت x برای میله‌های مربوط به مدل فعلی (i).
        # با اضافه کردن i * width، میله‌های مربوط به هر مدل کمی به سمت راست جابجا می‌شوند تا از هم جدا باشند.

    ax.set_ylabel('Score') #عنوان کلی برای محور y است که 'Score' را گذاشته.
    ax.set_title('Model Comparison')
    ax.set_xticks(x + width)
    #مکان قرارگیری نشانه‌های محور x تنظیم می‌شود.
    # از x + width استفاده می‌شود تا نشانه‌ها در وسط گروه‌های میله‌ها قرار بگیرند.

    ax.set_xticklabels(labels) #برچسب‌های مربوط به نشانه‌های محور x با استفاده از لیست labels تنظیم می‌شو
    ax.legend() #راهنمای نمودار
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    #یک تابع که به طور خودکار فاصله‌بندی بین عناصر نمودار (مانند عنوان، برچسب‌ها و راهنما) را تنظیم می‌کند تا از همپوشانی جلوگیری شود.

    plt.savefig(save_path)
    #نمودار تولید شده در فایلی با مسیری که در پارامتر save_path مشخص شده است، ذخیره می‌شود.

    plt.close()
    #شکل (figure) مربوط به نمودار بسته می‌شود تا منابع سیستم آزاد شوند.

    print(f"✅ نمودار ذخیره شد: {save_path}")


#  مقایسه مدل‌ها + تولید خروجی متنی و تصویری
def compare_models(X_test, y_test, models_dir):
    model_files = {
        "SVM": "svm_classifier.pkl",
        "Naive Bayes": "nb_classifier.pkl",
        "Logistic Regression": "lr_classifier.pkl"
    }

    results = []

    for model_name, filename in model_files.items():
        try:
            model_path = os.path.join(models_dir, filename)
            model = joblib.load(model_path)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            results.append((model_name, acc, prec, rec, f1))

            print(f"\n📊 {model_name} Scores:")
            print(f"Accuracy: {acc:.4f}")
            print(f"Precision: {prec:.4f}")
            print(f"Recall: {rec:.4f}")
            print(f"F1-Score: {f1:.4f}")

        except Exception as e:
            print(f"⚠️ خطا در بارگذاری یا ارزیابی {model_name}: {e}")

    #  مسیر خروجی‌ها: داخل src/evaluation_outputs
    base_dir = os.path.dirname(os.path.abspath(__file__))
    outputs_dir = os.path.join(base_dir, 'evaluation_outputs')
    os.makedirs(outputs_dir, exist_ok=True)

    #  گزارش متنی
    report_path = os.path.join(outputs_dir, 'compare_models_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Model Comparison:\n")
        f.write(f"{'Model':<20}{'Accuracy':<10}{'Precision':<10}{'Recall':<10}{'F1-Score':<10}\n")
        for model_name, acc, prec, rec, f1 in results:
            line = f"{model_name:<20}{acc:<10.2f}{prec:<10.2f}{rec:<10.2f}{f1:<10.2f}\n"
            f.write(line)
            print(line, end='')

    print(f"\n✅ گزارش متنی ذخیره شد: {os.path.abspath(report_path)}")

    # 🖼️ ذخیره نمودار
    chart_path = os.path.join(outputs_dir, 'compare_models_chart.png')
    plot_model_comparison(results, chart_path)

#  تابع main فقط برای تست (اختیاری)
def main():
    import pandas as pd
    from sklearn.model_selection import train_test_split

    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed_data.csv')
    df = pd.read_csv(data_path)

    X = df.drop('label', axis=1)
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    compare_models(X_test, y_test, models_dir)

if __name__ == "__main__":
    main()

"""
برای خروجی این فایل باید از فایل evaluate_models.py خروجی گرفته شود  
"""