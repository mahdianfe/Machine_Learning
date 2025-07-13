# src/main.py

import sys
import os  #برای کار با فایل‌ها و مسیرها
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer #تبدیل متن به ویژگی‌های عددی
#  در مدل ما، از TfidfVectorizer استفاده کردیم که مربوط به متن هستش و
#  مقدارهایی مثل 0.32، 0.05، ... تولید می‌کنه (یعنی غیرباینریه).
from sklearn.model_selection import train_test_split #جداسازی دیتا به تست و ترین
from sklearn.metrics import classification_report # ارزیابی مدل
import joblib #ذخیره مدل


# افزودن مسیر اصلی پروژه به sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# مقدار __file__ برابر با مسیر کامل فایل main.py خواهد بود
# در پایتون، .. یک نماد است که به پوشه‌ی والد (پوشه‌ای که در آن پوشه فعلی قرار دارد) اشاره می‌کند.
#  یعنی .. به این معناست که از پوشه جاری یک سطح بالا برو.
# در واقع، این خط به پایتون می‌گوید که از مسیر بالاتر (یعنی project/) شروع به جستجوی ماژول‌ها کند، نه فقط در پوشه‌ی main.py.

# ایمپورت کردن مدل‌ها به صورت داینامیک
model_name = sys.argv[1] if len(sys.argv) > 1 else "nb"  # پیش‌فرض: Naive Bayes
#با استفاده از sys.argv، از بیرون فایل هم می‌توانی مدل را تعیین کنی.
#این خط تعیین می‌کنه که مدل پیش‌فرض چی باشه، نه اینکه مدل واقعاً ایمپورت یا استفاده شده باشه تا اینکه به خطهای بعدی برسه.

# بدون این خط میشد به صورت دستی هر بار نام مدل را قرار بدیم مثلا
# model_name = "svm"
# اما در این صورت نمی‌تونی از بیرون (یعنی از طریق ترمینال) مدل رو تغییر بدی
# و مجبور می‌شی هر بار بیای توی فایل main.py و دستی کد رو عوض کنی.

# به زبان ساده می‌گوید:
#     اگه کاربر مدلی را وارد کرده (یعنی آرگومان دوم وجود دارد)، از آن استفاده کن.
#     وگرنه، مدل پیش‌فرض را بذار 'nb' (یعنی Naive Bayes).

# #sys.argv[0] → نام فایل اجرایی: 'main.py'
# sys.argv[1] → اولین آرگومان نام مدل ما است: 'svm'

 # چرا > 1؟
# چون همیشه:
#     sys.argv[0] → نام فایل پایتون هست.
#     پس اگر فقط برنامه اجرا بشه (بدون هیچ آرگومانی)، طول لیست sys.argv فقط ۱ هست:
# ['main.py']
# بنابراین:
#     وقتی طول sys.argv بیشتر از 1 باشه، یعنی حداقل یک آرگومان بعد از اسم فایل هم وارد شده.


if model_name == "nb":
    from models.train_nb_model import train_model
    #  اگر فایل train_nb_model.py وجود نداشته باشه
    # که پیش‌فرض هم همینه، اون‌وقت در زمان اجرا ارور می‌گیری (مثلاً: ModuleNotFoundError).
elif model_name == "svm":
    from models.train_svm_model import train_model
elif model_name == "lr":
    from models.train_lr_model import train_model
else:
    raise ValueError("Unsupported model")


# تابع برای خواندن ایمیل‌ها
# # تمام فایلهای متنی را میخواند و داخل لیستی میریزد
def load_emails_from_folder(folder):
    emails = []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        with open(path, 'r', encoding='latin-1') as f:
            emails.append(f.read())
    return emails
# خواندن ایمیل‌های اسپم و معمولیاز پوشه‌هایشان در data/ می‌خواند.
spam_emails = load_emails_from_folder('../data/spam')
ham_emails = load_emails_from_folder('../data/easy_ham')

# ساخت دیتافریم و دادن لیبل به انها
# # ستون text شامل متن ایمیل‌ها.
# # ستون label: عدد ۱ برای اسپم، عدد ۰ برای ایمیل‌های معمولی.
df = pd.DataFrame({'text': spam_emails + ham_emails,
                   'label': [1]*len(spam_emails) + [0]*len(ham_emails)})
# #فرض کن 3 ایمیل اسپم داریم
# # اون‌وقت len(spam_emails) می‌شه 3. حالا: [1] * 3
# # مساوی می‌شه با: [1, 1, 1]
# # # یعنی برای ۳ ایمیل اسپم، ۳ عدد ۱ تولید کردیم. همین کار برای ایمیل‌های معمولی:
# # [0] * len(ham_emails)
# # در نتیجه:
# # y = [1, 1, 1, 0, 0, 0]

#تبدیل محتوای ایمیل به عدد
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
# #می‌گوید که کلمات توقف (stop words) زبان انگلیسی را نادیده بگیرد.مانند "the"، "a"، "is"، "are" و غیره.
# با حذف این کلمات، تمرکز مدل بر روی کلمات مهم‌تر خواهد بود.
# #فقط ۱۰۰۰ کلمه‌ی مهم‌تر و پرتکرارتر را از بین تمام کلمات موجود در همه ایمیل‌ها نگه دار و بقیه را نادیده بگیر
X = vectorizer.fit_transform(df['text'])
#fit() یعنی: از روی تمام متن‌ها، کلمات پرتکرار و مهم (طبق TF-IDF) رو یاد بگیر
# و برای هر کلمه، یک عدد شاخص مشخص کن.
# transform() یعنی: حالا بیایم هر متن رو تبدیل به بردار عددی کنیم با توجه به آن چیزی که در fit() یاد گرفتیم.
y = df['label']

#دسته بندی دیتا به تست و ترین
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# آموزش مدل
model = train_model(X_train, y_train)

#  پیش‌بینی روی داده‌های تست و چاپ گزارش عملکرد مدل.
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

#

# # اطمینان از وجود فولدر models
os.makedirs('../models', exist_ok=True)

# ذخیره مدل و بردار‌ساز
joblib.dump(model, f'../models/{model_name}_classifier.pkl')
joblib.dump(vectorizer, '../models/vectorizer.pkl')


"""
مهم:
1. برای اجرا به فایل main میرویم
2. با توجه به اینکه خروجی چه مدلی را میخواهیم یکی از کدهای زیر را در ترمینال میزنیم
python main.py nb    
python main.py svm
python main.py lr

"""

#مدل‌های مختلف شما در پوشه models/ ذخیره خواهند شد.
# به عنوان مثال، اگر مدل SVM را انتخاب کنید، فایل مدل ذخیره‌شده به نام svm_classifier.pkl خواهد بود.

#########################################
# توضیح بیشتر کدهای بالا:

 # چرا برای انیکه مدلهای مختلفی را اموزش داده باشیم از تیکه کدهای بالا در مین استفاده میکنیم و نگفتیم model_name = "nb"؟
# مسلماً می‌توانی بنویسی:
# model_name = "nb"
# و کار هم می‌کند. اما این کار:
#     اجرای فایل را محدود به یک مدل خاص می‌کند.
#     اگر بخواهی مدل را عوض کنی، باید وارد کد شوی و دستی تغییر بدهی.
# در حالی‌که با sys.argv، از بیرون فایل هم می‌توانی مدل را تعیین کنی.
#نمونه اجرا
    # python main.py nb     # اجرای مدل Naive Bayes
    # python main.py svm    # اجرای مدل SVM
    # python main.py lr     # اجرای Logistic Regression



########################################
#
# # 🔹 نمایش چند پیش‌بینی مدل روی ایمیل‌های تست واقعی از دیتاست
# #
# # print("\n🔍 Sample predictions on test data:\n")
# #
# # # بازیابی ۵ ایمیل و نمایش پیش‌بینی مدل
# # for i in range(5):
# #     email_text = df['text'].iloc[y_test.index[i]]
# #     true_label = y_test.iloc[i]
# #     predicted_label = y_pred[i]
# #
# #     print(f"Email {i+1}:")
# #     print(email_text[:200].replace('\n', ' '))  # فقط ۲۰۰ کاراکتر اول برای نمایش
# #     print("Prediction:", "SPAM ❌" if predicted_label == 1 else "NOT SPAM ✅")
# #     print("Actual:", "SPAM ❌" if true_label == 1 else "NOT SPAM ✅")
# #     print("-" * 60)
