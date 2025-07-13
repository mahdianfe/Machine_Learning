# 🛡️ پروژه تشخیص تقلب در تراکنش‌های کارت اعتباری – P2_Credit_Card_Fraud_Detection

این پروژه یک سیستم تشخیص تقلب در تراکنش‌های کارت اعتباری را با استفاده از تکنیک‌های یادگیری ماشین طراحی و پیاده‌سازی می‌کند. هدف این است که بتوانیم تراکنش‌های مشکوک را با دقت بالا از بین تراکنش‌های عادی تشخیص دهیم.

---

## 🧠 خلاصه تکنیکی پروژه

- استفاده از Logistic Regression و امکان توسعه برای مدل‌های پیچیده‌تر مانند Random Forest و XGBoost
- اعتبارسنجی داده‌ها (کیفیت داده، حذف مقادیر پرت و نامعتبر)
- پیش‌پردازش و نرمال‌سازی (Scaling، حذف Outlierها، Drop NA)
- مهندسی ویژگی‌ها (Feature Engineering)
- تقسیم داده‌ها به مجموعه آموزش و تست
- آموزش و ارزیابی مدل‌ها با متریک‌های کامل (Accuracy, F1, ROC AUC و ...)
- رسم منحنی ROC برای مقایسه عملکرد مدل‌ها
- تنظیم هایپرپارامترها با GridSearchCV
- تشخیص ناهنجاری‌ها با Isolation Forest
- ذخیره مدل‌ها و متریک‌ها در فایل‌های قابل استفاده مجدد

---

## 🗂 ساختار پروژه

```bash
P2_credit_card_fraud_detection_project/
├── data/
│   ├── raw/
│   │   └── creditcard.csv
│   └── processed/
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
├── models/
├── notebooks/
│   ├── data_analysis.ipynb
│   └── model_training_evaluation.ipynb
├── src/
│   ├── main.py
│   ├── pipeline.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   ├── hyperparameter_tuning.py
│   ├── anomaly_detection.py
│   ├── feature_engineering.py
│   └── data/
│       ├── data_load.py
│       ├── data_preprocess.py
│       ├── data_validate.py
│       └── data_split.py
├── outputs/
│   ├── Logistic Regression_model.joblib
│   ├── model_performance_metrics.csv
│   ├── model_performance_metrics.txt
│   ├── metrics_comparison.png
│   └── roc_curves.png
├── requirements.txt
└── README.md
```

---

## 🚀 مراحل اجرای پروژه

### 1. 📥 بارگذاری داده
از فایل CSV در مسیر `data/raw/creditcard.csv` با استفاده از `pandas.read_csv`.

### 2. ✅ اعتبارسنجی داده
بررسی موارد زیر:
- نبود مقادیر NaN
- فقط شامل مقادیر 0 و 1 در ستون `Class`
- مثبت بودن `Amount`
- منطقی بودن مقادیر `Time` (مثلاً زیر 48 ساعت)

### 3. 🧼 پیش‌پردازش داده
- حذف مقادیر پرت در `Amount` با `clipping`
- نرمال‌سازی `Amount` و `Time` با `StandardScaler`
- حذف ردیف‌های تکراری

### 4. 🔬 مهندسی ویژگی
- نرمال‌سازی کل ویژگی‌ها
- جدا کردن `X` و `y`

### 5. ✂️ تقسیم داده و ذخیره
- تقسیم با `train_test_split`
- ذخیره در `data/processed/`

### 6. 🧱 ساخت پایپ‌لاین
- `Pipeline(StandardScaler → Classifier)`

### 7. 🏋️ آموزش مدل
با استفاده از `LogisticRegression` (و قابل توسعه)

### 8. 📊 ارزیابی مدل با Cross Validation
- محاسبه متریک‌ها روی ۵ فولد
- ذخیره در `cv_metrics.csv`

### 9. 📈 رسم منحنی ROC
با استفاده از `matplotlib` برای همه مدل‌ها

### 10. ⚙️ تنظیم هایپرپارامترها
- با `GridSearchCV` روی `RandomForestClassifier`

### 11. 🚨 تشخیص ناهنجاری
- با `IsolationForest` روی داده‌های تست

### 12. 💾 ذخیره مدل‌ها و متریک‌ها
- خروجی مدل‌ها در `outputs/`
- ذخیره `.joblib`, `.csv`, `.txt`, `.png`

---

## ⚙️ نحوه اجرا

### 1. نصب وابستگی‌ها
```bash
pip install -r requirements.txt
```

### 2. اجرای کامل پروژه
```bash
python src/main.py
```

---

## 📊 کتابخانه‌های کلیدی استفاده‌شده

- `pandas`, `numpy` برای پردازش داده
- `scikit-learn` برای مدل‌سازی و ارزیابی
- `matplotlib`, `seaborn` برای مصورسازی
- `joblib` برای ذخیره‌سازی مدل

---

## 📊 نوت‌بوک‌ها

- `notebooks/data_analysis.ipynb`: تحلیل اکتشافی داده‌ها (EDA)
- `notebooks/model_training_evaluation.ipynb`: آموزش مدل و تحلیل نتایج

---

## 🧳 منبع داده

- [Kaggle – Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)

---

## 📜 مجوز

این پروژه تحت مجوز MIT ارائه شده است. برای مشاهده جزئیات به فایل `LICENSE` مراجعه نمایید.

---

## 🤝 مشارکت

مشارکت شما باعث خوشحالی است!

1. این پروژه را Fork کنید
2. روی Branch خود توسعه دهید
3. Pull Request ارسال کنید

---

## ☕ حمایت از توسعه‌دهنده

اگر این پروژه براتون مفید بوده، یک فنجان قهوه لطف بزرگیه:

[خریدن یک فنجان قهوه برای من](https://www.coffeebede.com/mahdianfe)

---

**ساخته‌شده با 💻، عشق و مقدار زیادی ☕ برای مقابله با تقلب در کارت‌های اعتباری!**
