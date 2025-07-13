# 📧 تشخیص‌دهنده اسپم ایمیل

☕ برای اینکه رفرش بشم، لطفا یه کافی میدی؟  
[خریدن یه فنجون قهوه](https://www.coffeebede.com/mahdianfe)

یک سیستم یادگیری ماشین برای تشخیص اسپم که ایمیل‌ها را به دو دسته "اسپم" و "غیر اسپم (Ham)" طبقه‌بندی می‌کند.

## 🚀 ویژگی‌ها

- پیش‌پردازش ایمیل‌های خام (شامل هدرها و متن ایمیل)
- استخراج ویژگی‌های مهم (Bag-of-Words، TF-IDF و غیره)
- آموزش با الگوریتم‌های مرسوم یادگیری ماشین مثل Naive Bayes و SVM
- ارزیابی با دقت، بازیابی و صحت
- قابلیت اجرای خط فرمان برای تست ایمیل‌های جدید


## 📂 ساختار پروژه

```
P1_email_spam_detector/
├── README_En.md             
├── README_Fa.md             
├── requirements.txt         
├── test_emails.txt  
├── .idea/    
│   ├── .gitignore                               
│   ├── inspectionProfiles/
│   ├── modules.xml
│   └── P1_email_spam_detector.iml
│   └── workspace.xml
├── data/
│   ├── easy_ham/
│   ├── spam/
│   ├── 20021010_easy_ham.tar.bz2
│   └── 20021010_spam.tar.bz2
├── models/                     
│   ├── __init__.py
│   ├── lr_classifier.pkl
│   ├── nb_classifier.pkl
│   ├── svm_classifier.pkl
│   ├── train_lr_model.py
│   ├── train_nb_model.py
│   ├── train_svm_model.py
│   └── vectorizer.pkl
├── notebooks/                  
│   ├── __init__.py
│   ├── explore_data.py         
│   ├── step01_spam_email_basic_notebook.ipynb
│   └── step02_spam_email_structured_version.ipynb
└── src/                        
    ├── evaluation_outputs/     
    ├── __init__.py             
    ├── compare_models.py
    ├── evaluate_lr_model.py
    ├── evaluate_models.py
    ├── evaluate_nb_model.py
    ├── evaluate_svm_model.py
    ├── main.py                 
    ├── predict.py
    ├── predict_email.py
    ├── predict_from_file.py
    └── preprocess.py
```

## 🛠️ نصب و اجرا

```bash
git clone https://github.com/yourusername/P1_email_spam_detector.git
cd P1_email_spam_detector
python -m venv venv
source venv/bin/activate   # در ویندوز: venv\\Scripts\\activate
pip install -r requirements.txt
```

## 🧪 نحوه استفاده

### آموزش مدل:
```bash
python main.py --train
```

### تست روی ایمیل‌های جدید:
```bash
python main.py --test test_emails.txt
```

## 🧠 تکنولوژی‌های استفاده‌شده

- Python 3.x
- Scikit-learn
- Numpy / Pandas
- NLTK / Regex

## 🧳 منبع دیتاست

- [SpamAssassin Public Corpus](http://spamassassin.apache.org/old/publiccorpus/)

## 📜 لایسنس

MIT License

## 🤝 مشارکت

در صورت تمایل به مشارکت، pull request خوشحال خواهیم بود.

---
*ساخته‌شده با 💻 و ☕ برای یادگیری و مقابله با اسپم!*
