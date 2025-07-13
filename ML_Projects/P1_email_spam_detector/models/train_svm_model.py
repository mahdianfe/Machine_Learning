# models/train_svm_model.py

from sklearn.svm import SVC

def train_model(X_train, y_train):
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    return model


"""
مهم:
1. برای اجرا به فایل main میرویم
2. کد زیر را در ترمینال میزنیم و به صورت مستقیم علامت پلی را نیمزنیم
python main.py svm

"""

#مدل‌های مختلف شما در پوشه models/ ذخیره خواهند شد.
# به عنوان مثال، اگر مدل SVM را انتخاب کنید، فایل مدل ذخیره‌شده به نام svm_classifier.pkl خواهد بود.


