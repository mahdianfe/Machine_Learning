# models/train_nb_model.py
#################################
# Training code goes here
#🔹 معمولاً آموزش مدل از اینجا جدا می‌شود،
# مثلاً اگر بخواهید فقط مدل را دوباره آموزش دهید بدون اجرای بقیه چیزها.
# 🔹 بهتره برای هر مدل یک فایل جدید آموزشی داشته باشی، مثل:
#     train_nb.py (برای Naive Bayes)
#     train_lr.py (برای Logistic Regression)
#     train_svm.py (برای SVM)

#################################
# train_nb_model.py
from sklearn.naive_bayes import MultinomialNB

def train_model(X_train, y_train):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model


"""
مهم:
1. برای اجرا به فایل main میرویم
2. کد زیر را در ترمینال میزنیم و به صورت مستقیم علامت پلی را نیمزنیم
python main.py nb

"""



