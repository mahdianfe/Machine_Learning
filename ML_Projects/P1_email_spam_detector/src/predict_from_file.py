# src/predict_from_file.py

from joblib import load
from predict import predict_email


def predict_emails_from_file(filepath, model_name):
    # خواندن محتوای فایل
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # تقسیم ایمیل‌ها
    emails = content.split('---')

    # پیش‌بینی برای هر ایمیل
    for i, email in enumerate(emails):
        email = email.strip()
        if not email:
            continue

        # استفاده از تابع predict_email برای پیش‌بینی ایمیل
        result = predict_email(model_name, email)

        # نمایش نتیجه
        print(f"\nEmail {i + 1}:")
        print(email)
        print("Result:", result)


if __name__ == "__main__":
    model_name = 'svm'  # مدل مورد نظر را انتخاب کنید: 'svm', 'nb', یا 'lr'
    predict_emails_from_file("test_emails.txt", model_name)

## ____________________________________________
# برای اجرای کد:
# در ترمینال وارد پوشه پروژه شو و بنویس:
"""
python src/predict_from_file.py
"""
# چون وقتی با دکمه ▶️ پلی اجرا می‌کنی، PyCharm فایل رو اجرا می‌کنه ولی دایرکتوری جاری (Current Working Directory) ممکنه پوشه‌ی src باشه.
# فایل‌های مدل شما (models/spam_classifier.pkl) در پوشه‌ی بالاتر هستند، بنابراین پیدا نمی‌شن.
