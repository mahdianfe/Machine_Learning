
# 📧 Email Spam Detector

☕ Enjoyed this project? Consider buying me a coffee to keep me energized!
[Please give me a coffee.](https://www.coffeebede.com/mahdianfe)

A machine learning-based email spam detection system that classifies emails as "spam" or "ham" (non-spam) using a preprocessed dataset of real-world emails.

## 🚀 Features

- Preprocessing of raw email data (including headers and body text)
- Extraction of relevant features (bag-of-words, TF-IDF, etc.)
- Training using classic ML models (e.g. Naive Bayes, SVM)
- Evaluation metrics like accuracy, precision, recall
- CLI or script-based classification of new emails

## 📂 Project Structure

```
تصویر و توضیحات شما نشان می‌دهد که ساختار پروژه P1_email_spam_detector شما در حال حاضر تفاوت‌های قابل توجهی با ساختار ایده‌آل قبلی دارد، به ویژه اینکه بخش عمده‌ای از فایل‌ها به جای ریشه پروژه، در پوشه src قرار گرفته‌اند.

هدف ما این است که ساختار فعلی را به چیزی شبیه به ساختار مورد نظر شما نزدیک کنیم و توضیح دهیم که هر بخش چه کاربردی دارد.

تحلیل ساختار فعلی (بر اساس عکس):

پوشه ریشه (P1_email_spam_detector):

    .idea/: پوشه مربوط به تنظیمات IDE (احتمالاً PyCharm). این نباید در Git باشد.

    .gitignore: خوب است، برای نادیده گرفتن فایل‌ها/پوشه‌های ناخواسته.

    modules.xml, P1_email_spam_detector.iml, workspace.xml: فایل‌های IDE، نباید در Git باشند.

    data/: شامل easy_ham و spam و فایل‌های فشرده. این بخش خوب است.

    models/: شامل *.pkl و *.py. این مدل‌ها و اسکریپت‌های مربوط به مدل‌ها هستند. این بخش هم خوب است.

    notebooks/: شامل فایل‌های Jupyter Notebook. این هم خوب است.

    src/: اینجا مشکل اصلی است. بسیاری از فایل‌های اصلی پروژه که باید در ریشه باشند، اینجا قرار گرفته‌اند:

        evaluation_outputs/

        _init__.py

        compare_models.py

        evaluate_lr_model.py

        evaluate_models.py

        evaluate_nb_model.py

        evaluate_svm_model.py

        main.py

        predict.py

        predict_email.py

        predict_from_file.py

        preprocess.py

        _init__.py

        README_spam_detector_En (با وضعیت MU یعنی Modified, Untracked)

        README_spam_detector_Fa (با وضعیت MU یعنی Modified, Untracked)

        requirements.txt

        test_emails.txt

ساختار پروژه اصلاح شده و پیشنهادی (بر اساس خواسته شما و Best Practices):

با توجه به خواسته شما و برای داشتن یک پروژه پایتون منظم‌تر، ساختار پیشنهادی به شکل زیر است:

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

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/P1_email_spam_detector.git
cd P1_email_spam_detector

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`

# Install required packages
pip install -r requirements.txt
```

## 🧪 How to Use

### Train the Model:
```bash
python main.py --train
```

### Test on New Emails:
```bash
python main.py --test test_emails.txt
```

## 🧠 Technologies Used

- Python 3.x
- Scikit-learn
- Numpy / Pandas
- NLTK / Regex for text preprocessing

## 🧳 Dataset Source

- The [SpamAssassin Public Corpus](http://spamassassin.apache.org/old/publiccorpus/)

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

*Made with 💻 and ☕ for learning and detecting spam!*
