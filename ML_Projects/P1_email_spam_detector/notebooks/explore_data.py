"""دید کلی از دیتاست"""
import pandas as pd

# #_____________________
# """هر جا که نیاز به لود کردن فایل (مدل یا فایل‌های دیگر) از مسیرهای پروژه‌ت داری،
#  این بلوک مسیرسازی رو در ابتدای فایل استفاده کن."""
# import os
# from joblib import load
#
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # پوشه ریشه پروژه
# MODEL_PATH = os.path.join(BASE_DIR, 'models', 'spam_classifier.pkl')
#
# model = load(MODEL_PATH)
# # یعنی:
# #     __file__: مسیر فایل جاری (مثلاً src/predict.py)
# #     os.path.abspath(__file__): تبدیلش به مسیر کامل (مثلاً: C:/projects/email-spam-detector/src/predict.py)
# #     os.path.dirname(...) یک پوشه بالا میره → یعنی C:/projects/email-spam-detector
# #     پس: BASE_DIR یعنی پوشه ریشه پروژه
# #_____________________

# explore_data.py
"""دید کلی از دیتاست"""
import pandas as pd
import os  # برای کار با فایل ها

# 🔹 تابع برای خواندن ایمیل ها از پوشه
def load_emails_from_folder(folder):
    emails = []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        with open(path, 'r', encoding='latin-1') as f:
            emails.append(f.read())
    return emails

# 🔹 تابع برای بارگیری داده‌های ایمیل (ترکیب اسپم و غیر اسپم)
def load_email_data():
    spam_emails = load_emails_from_folder('data/spam')
    ham_emails = load_emails_from_folder('data/easy_ham')
    return spam_emails, ham_emails

# بارگیری داده‌ها
spam_emails, ham_emails = load_email_data()

# ایجاد دیتافریم برای ایمیل‌های هرزنامه
spam_df = pd.DataFrame({'text': spam_emails, 'label': 1})

# ایجاد دیتافریم برای ایمیل‌های غیر هرزنامه
ham_df = pd.DataFrame({'text': ham_emails, 'label': 0})

# ترکیب دو دیتافریم
df = pd.concat([spam_df, ham_df], ignore_index=True)

print("------- First few rows of the combined DataFrame: --------")
print(df.head())

print("\n ------- Combined DataFrame information: -------")
print(df.info())

print("\n------- Number of spam and non-spam emails: -------")
print(df['label'].value_counts())

# نمایش چند نمونه از متن ایمیل
print("\n------- Email text samples: -------")
for i in range(3):
    print(f"\n------- sample {i+1} (lable: {df['label'][i]}): -------")
    print(df['text'][i][:200], "...")

#____________________
# برای اجرای این فایل، کد زیر را در ترمینال بزنید
"""
python notebooks/explore_data.py
"""