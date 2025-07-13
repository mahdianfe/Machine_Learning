# src/predict_email.py
# # 🔹 بعد از ساخت و اموزش و کارهای اصلی
# # این فایل جدید را ساختیم تا برایمان بگوید که ایمیل های دستی زیر اسپم هستند یا خیر
# #         "Congratulations! You won a free ticket to Bahamas!",
# #         "Hi John, can we meet at the cafe tomorrow?",
# #         "URGENT: Update your banking information immediately.",
# #         "Here is the report you asked for. Let me know your thoughts."

##______________________
# # توضیح بیشتر:
# # آیا نمی‌شد فایلهای   predict_email.py و  predict.py را یکی کرد؟
# #
# # 1. فایل predict.py قراره مثل مغزِ پیش‌بینی مدل باشه. می‌تونه توسط Flask، تست، یا هر اسکریپت دیگه import بشه.
# # ولی predict_email.py فقط یه تست‌کننده ساده است. فقط برای انسان‌ها، نه ماشین یا سیستم.
#
# # 2اگر این دو فایل را یکی کنیم از نظر طراحی پروژه اشتباه است. اگه این کارو بکنی:
# #     هر بار که یه فایل کدی import predict کنه، کل اون ایمیل‌های تست هم اجرا می‌شن! ❌
# #     فایل سنگین، درهم، و غیرقابل نگهداری میش



# predict_email.py

from predict import predict_email

if __name__ == '__main__':
    emails = [
        "Congratulations! You won a free ticket to Bahamas!",
        "Meeting confirmed at 3pm with the HR team.",
        "Claim your prize now! Click here."
    ]

    model_name = 'svm'  # می‌توانید مدل را به 'svm', 'nb', یا 'lr' تغییر دهید.

    for i, email in enumerate(emails):
        print(f"Sample {i+1}:\n{email}")
        print(predict_email(model_name, email))
        print("-" * 30)

#____________________________________________
# # برای اجرای کد:
# # در ترمینال وارد پوشه پروژه شو و بنویس:
"""
 python src/predict_email.py
 
 """
# # چون وقتی با دکمه ▶️ پلی اجرا می‌کنی، PyCharm فایل رو اجرا می‌کنه
# ولی دایرکتوری جاری (Current Working Directory) ممکنه پوشه‌ی src باشه.
# # فایل‌های مدل شما (models/spam_classifier.pkl) در پوشه‌ی بالاتر هستند، بنابراین پیدا نمی‌شن.
