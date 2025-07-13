# src/evaluate_models.py
# ارزیابی مدل
##################
from evaluate_svm_model import evaluate_svm_model
from evaluate_nb_model import evaluate_nb_model
from evaluate_lr_model import evaluate_lr_model
import os
from sklearn.model_selection import train_test_split
import pandas as pd
from compare_models import compare_models
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
# ___________________________________

# مسیر نسبی به فولدر data
base_dir = os.path.dirname(os.path.abspath(__file__))
#os.path.abspath(__file__): این قسمت، مسیر مطلق (کامل) فایل جاری (evaluate_models.py) را در سیستم عامل برمی‌گرداند.

spam_dir = os.path.join(base_dir, '..', 'data', 'spam')
#os.path.join(...): این تابع برای ساختن یک مسیر فایل یا دایرکتوری به صورت هوشمندانه و سازگار با سیستم عامل‌های مختلف استفاده می‌شود. این تابع به طور خودکار از جداکننده مناسب مسیر (مثل / در لینوکس و \ در ویندوز) استفاده می‌کند.
# '..': این علامت در مسیر به معنای "رفتن به دایرکتوری والد" است.
# 'data', 'spam', 'easy_ham': اینها نام دایرکتوری‌ها یا فایل‌ها هستند.

ham_dir = os.path.join(base_dir, '..', 'data', 'easy_ham')

# تابع برای خواندن ایمیل‌ها
def load_emails_from_folder(folder):
    emails = []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        with open(path, 'r', encoding='latin-1') as f:
            emails.append(f.read())
    return emails

# خواندن ایمیل‌ها
spam_emails = load_emails_from_folder(spam_dir)
ham_emails = load_emails_from_folder(ham_dir)
df = pd.DataFrame({'text': spam_emails + ham_emails,
                   'label': [1]*len(spam_emails) + [0]*len(ham_emails)})


models_dir = os.path.join(base_dir, '..', 'models')
vectorizer_path = os.path.join(models_dir, 'vectorizer.pkl')

# تقسیم داده‌ها به آموزش و تست بدون فیت
vectorizer = joblib.load(vectorizer_path)
X = vectorizer.transform(df['text'])   # فقط transform (بدون fit)
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ___________________________________

def main():
    try:
        print("Evaluating SVM Model:")
        svm_path = os.path.join(models_dir, 'svm_classifier.pkl')
        model_svm = joblib.load(svm_path)
        evaluate_svm_model(model_svm, X_test, y_test)
    except Exception as e:
        print(f"Error evaluating SVM model: {e}")

    try:
        print("\nEvaluating Naive Bayes Model:")
        nb_path = os.path.join(models_dir, 'nb_classifier.pkl')
        model_nb = joblib.load(nb_path)
        evaluate_nb_model(model_nb, X_test, y_test)
    except Exception as e:
        print(f"Error evaluating Naive Bayes model: {e}")

    try:
        print("\nEvaluating Logistic Regression Model:")
        lr_path = os.path.join(models_dir, 'lr_classifier.pkl')
        model_lr = joblib.load(lr_path)
        evaluate_lr_model(model_lr, X_test, y_test)
    except Exception as e:
        print(f"Error evaluating Logistic Regression model: {e}")

    print("\nComparing all models:")
    compare_models(X_test, y_test, models_dir)  # اینجا منتقل شود داخل main



#_________________________________________

# مسیر فولدر evaluation_outputs
outputs_dir = "evaluation_outputs"

# اگر فولدر evaluation_outputs وجود نداشت، آن را ایجاد کن
if not os.path.exists(outputs_dir):
    os.makedirs(outputs_dir)
    print(f"Folder '{outputs_dir}' created.")
else:
    print(f"Folder '{outputs_dir}' already exists.")


if __name__ == "__main__":
    main()


## ____________________________________________
"""
1. این فایل را میتوان با دکمه پلی خروجی گرفت
2. این فایل خروجی فایلهای زیر را میگیرد و در پوشه src/evaluation_outputs ذخیره میکند
evaluate_lr_model.py و evaluate_nb_model و evaluate_svm_model.py و compare_models.py  
"""