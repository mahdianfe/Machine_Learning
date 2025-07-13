
# # مربوط به گام 10

# src/hyperparameter_tuning.py
import os
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def tune_hyperparameters(X_train, y_train, output_dir="outputs", cv=3):
    """
    روی سه مدل مختلف سرچ شبکه‌ای انجام می‌دهد و
    نتایج را در یک CSV ذخیره می‌کند.
    """
    os.makedirs(output_dir, exist_ok=True)

    search_space = {
        "Logistic Regression": {
            "estimator": LogisticRegression(max_iter=1000, class_weight='balanced', tol=0.01),
            # tol تعیین می‌کند که این فرآیند تکراری چه زمانی متوقف شود. اگر بهبود در تابع هدف یا تغییر در وزن‌های مدل در یک گام، از مقدار tol کمتر شود، الگوریتم بهینه‌سازی فرض می‌کند که به همگرایی (Convergence) رسیده است و متوقف می‌شود.
            # tol بزرگتر (مثلاً 0.001 یا 0.01): یعنی الگوریتم می‌تواند زودتر متوقف شود، حتی اگر هنوز کمی جای بهبود وجود داشته باشد.


            "param_grid": {
                "C":       [0.1, 1.0, 10.0],
                # مثل SVM، C در Logistic Regression هم همان ضریب regularization است.
                # C بزرگ‌تر → مدل بیشتر روی درستی تمرکز می‌کنه
                # C کوچک‌تر → مدل ساده‌تر و پایدارتر

                "solver":  ["liblinear"]
            #     مشخص می‌کنه که الگوریتم عددی برای بهینه‌سازی چه باشد
            # مثلاً اگر L1 Penalty بخوای، باید liblinear استفاده کنی. ولی اگر تعداد ویژگی‌ها زیاد باشه، lbfgs سریع‌تره.
            #     هدف GridSearch اینه که تست کنه کدوم solver با کدوم مقدار C، بهترین نتیجه رو می‌ده
            # نکته مهم:
            #     همه‌ی solverها با همه‌ی مقدارهای C یا penaltyها سازگار نیستن.
                # در این مثال چون فقط از L2 استفاده می‌کنی (که پیش‌فرضه)، هر دو solver (liblinear, lbfgs) معتبر هستن.
                # اگر مثلاً از penalty='l1' استفاده می‌کردی، دیگه lbfgs پشتیبانی نمی‌کرد.
            }
            #  پارام گرید یک دیکشنری است که می‌گوید: «برای هر هایپرپارامتر چه مقادیری را امتحان کن».

        },
        "Random Forest": {
            "estimator": RandomForestClassifier(),
            "param_grid": {
                "n_estimators": [100, 200],
                "max_depth":    [10, 20, None]
            }
        #     جنگل تصادفی درخت می‌سازد، پس «تعداد درخت‌ها» (n_estimators) و «عمق درخت» (max_depth) معنی‌دار است.
        },
        "SVM": {
            "estimator": SVC(probability=True),
            # SVC(kernel="linear", probability=False) #جایگزین خط بالا شود بخاطر طولانی شدن روند خروجی
            "param_grid": {
                "C":     [0.1, 1.0, 10.0],
                # #C ضریبِ معکوسِ regularization است.
                #  «مدل را با سه درجه‌ی مختلف regularization امتحان کن تا ببینیم کدام بهتر جواب می‌دهد.»
                # بزرگ‌تر شدن C ⇒ مجازات خطا کمتر ⇒ مدل منعطف‌تر (ریسک overfit بالاتر).
                # کوچک شدن C ⇒ مجازات خطا بیشتر ⇒ مدل ساده‌تر (ریسک underfit).
                # 🔁 GridSearch این ۳ مقدار را تست می‌کند و با Cross Validation می‌سنجد کدام مقدار C دقت یا AUC بیشتری می‌دهد.

                "kernel":["linear"]
                #هسته خطی (Linear Kernel) ساده‌ترین نوع هسته است. این هسته فرض می‌کند که داده‌های شما از ابتدا در فضای ویژگی (Feature Space) خود به صورت خطی قابل جداسازی هستند.
            }
        #     SVM خط مرزی یا کرنل می‌سازد، پس «نوع هسته» (kernel) و «C» مهم‌اند.
        }
    }


    all_results = []

    for name, cfg in search_space.items():
        print(f"🔍  شروع GridSearch برای «{name}» …")

        X_sample = X_train.sample(n=5000, random_state=42)
        y_sample = y_train.loc[X_sample.index]
        # به جای استفاده از کل X_train که بسیار بزرگ است، می‌تونی فقط بخشی از داده‌ها رو به GridSearch بدهی. تا خروجی خیلی طولانی نشود

        grid = GridSearchCV(
            estimator = cfg["estimator"],
            # به GridSearchCV می‌گوید که از کدام مدل پایه (مثلاً LogisticRegression()) برای آموزش و ارزیابی استفاده کند.
            # این مدل هنوز آموزش ندیده و فقط به عنوان الگو (template) برای GridSearchCV داده می‌شود.

            param_grid= cfg["param_grid"],
            # یک دیکشنری است که می‌گوید: «برای هر هایپرپارامتر چه مقادیری را امتحان کن».

            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,            # استفاده از همۀ هسته‌ها
            verbose=1             # نمایش پیشرفت روی ترمینال
        )
        # این قسم از کد یک شیء از کلاس GridSearchCV را ایجاد می‌کند.
        # GridSearchCV مسئول جستجوی جامع و سیستماتیک تمام ترکیبات ممکن هایپرپارامترها در param_grid و پیدا کردن بهترین مدل است.

        # grid.fit(X_train, y_train)
        grid.fit(X_sample, y_sample)
        # از تمام داده ها استفاده نمیکنیم تا زمان خروجی کمتر شود

        print(f"✅  بهترین پارامترهای {name}: {grid.best_params_}")
        print(f"📈  بهترین ROC‑AUC: {grid.best_score_:.3f}\n")

        all_results.append({
            "Model":         name,
            "Best Params":   grid.best_params_,
            "Best ROC AUC":  grid.best_score_
        })

        # اگر بخواهی بهترین مدل را ذخیره کنی:
        best_path = os.path.join(output_dir, f"best_{name.replace(' ','_')}.joblib")
        import joblib; joblib.dump(grid.best_estimator_, best_path)

    # ذخیرۀ تمام نتایج در CSV
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(output_dir, "hyperparam_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"📄  نتایج کامل در «{csv_path}» ذخیره شد.")
