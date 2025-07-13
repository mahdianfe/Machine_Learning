# models/train_lr_model.py
from sklearn.linear_model import LogisticRegression

def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

"""
مهم:
1. برای اجرا به فایل main میرویم
2. کد زیر را در ترمینال میزنیم و به صورت مستقیم علامت پلی را نیمزنیم
python main.py lr

"""



