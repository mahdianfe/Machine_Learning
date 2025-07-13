# src/preprocess.py
# فقط برای transform داده
#####################################
# Preprocessing code goes here
#🔹 این فایل برای کارهای پیش‌پردازش (مثلاً پاک‌سازی داده‌ها، حذف علامت‌ها و ...).
# اما در این پروژه من از TfidfVectorizer با پارامتر stop_words='english' در فایل مین استفاده کردم
# که کارهایی مثل حذف کلمات متداول (the, is, of...) را انجام می‌ده.
# اما نرمال‌سازی کامل‌تر توصیه می‌شه مثل:
#     lowercase کردن همه متن‌ها
#     حذف punctuation (.,!?" و غیره)
#     حذف اعداد
#     حذف spaceهای اضافی
#_________________________________

# preprocess.py

# import os
# from sklearn.feature_extraction.text import TfidfVectorizer
#
# def load_emails_from_folder(folder):
#     emails = []
#     for filename in os.listdir(folder):
#         path = os.path.join(folder, filename)
#         with open(path, 'r', encoding='latin-1') as f:
#             emails.append(f.read())
#     return emails
#
# def preprocess_data(spam_folder, ham_folder):
#     spam_emails = load_emails_from_folder(spam_folder)
#     ham_emails = load_emails_from_folder(ham_folder)
#
#     vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
#     X = vectorizer.fit_transform(spam_emails + ham_emails)
#     y = [1] * len(spam_emails) + [0] * len(ham_emails)
#
#     return X, y, vectorizer
