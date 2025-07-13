# مربوط به گام 8 و 9
# src/model_evaluation.py
import os
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.metrics import (
    make_scorer, accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns

# ──────────────────────────────
# 1)  تابع اصلی ارزیابی
# ──────────────────────────────
def evaluate_models_cv(models, X, y, output_dir="outputs", cv=5):
    """
    Cross-validation روی تمام مدل‌ها، ذخیره متریک‌ها و رسم نمودار مقایسه‌ای.
    """
    os.makedirs(output_dir, exist_ok=True)

    scorers = {
        "accuracy":  make_scorer(accuracy_score),
        "f1":        make_scorer(f1_score),
        "precision": make_scorer(precision_score),
        "recall":    make_scorer(recall_score),
        "roc_auc":  "roc_auc"
    }
    # با استفاده از make_scorer داریم:
    # به تابع امتیاز دهی (مانند accuracy_score ،precision ،recall ، f1 و.. ) ورودی های y_true و y_pred  (را که نیاز دارد ) را می دهیم و
    # آنرا به فرمتی استانداد تبدیل میکنیم تا در توابع ارزیابی متقابل (cross-validation) مانند cross_validate قابل استفاده باشد.
    # نکته2: در scorers مقدار اکیورسی و F1 و... محاسبه نمی‌شود صرفاً "تعریف می‌کند"

    results = []
    for name, model in models.items():
        print(f"📊 ارزیابی مدل: {name}")

        # پاارمترscoring
        scores = cross_validate(model, X, y, cv=cv, scoring=scorers, n_jobs=-1)
        # scores = cross_validate(...): این خط، تمام فرآیند Cross-validation را برای یک مدل خاص (model) انجام می‌دهد.
        # یعنی اگر cv=5 باشد، cross_validate مدل را 5 بار آموزش داده و ارزیابی می‌کند.
        # که در هر مرحله هر یک از scorers را هم برای هر مدل محاسبه میکند
        # در پایان این خط، متغیر scores یک دیکشنری حاوی تمام نتایج (مثلاً 5 مقدار برای دقت، 5 مقدار برای F1 Score و...) از این 5 تکرار خواهد بود.
        # این مرحله تا زمانی که scores به طور کامل پر نشده باشد (یعنی 5 تکرار انجام نشده باشد)، به خط بعدی نمی‌رود.
        # نکته : پارامتر n_jobs در تابع cross_validate (و بسیاری دیگر از توابع Scikit-learn که از پردازش موازی پشتیبانی می‌کنند) برای کنترل تعداد هسته‌های CPU یا "job" هایی استفاده میشه که به صورت موازی برای اجرای عملیات‌ها به کار گرفته میشن.
        #
        # مقدار    n_jobs = -1: این مقدار ویژه به cross_validate میگه که "تمام هسته‌های CPU موجود رو استفاده کن". این معمولاً بهترین گزینه برای سریع‌تر کردن فرآیند cross-validation، به خصوص در مجموعه داده‌های بزرگ یا وقتی cv (تعداد fold ها) بالا باشه، هست.
        #   مقدار  n_jobs = 1: به این معنیه که فقط یک هسته CPU استفاده میشه (اجرای سریالی).
        #    مقدار n_jobs = N (مثلاً 2 یا 4): به این معنیه که N هسته CPU به صورت موازی کار می‌کنند.

        res = {
            "Model":     name,
            "Accuracy":  scores["test_accuracy"].mean(),
            "F1 Score":  scores["test_f1"].mean(),
            "Precision": scores["test_precision"].mean(),
            "Recall":    scores["test_recall"].mean(),
            "ROC AUC":   scores["test_roc_auc"].mean()
        }
        # res = { ... }: پس از اینکه scores به طور کامل آماده شد، این خط اجرا می‌شود.
        # در این مرحله، مقادیر میانگین‌گرفته شده (مثل scores["test_accuracy"].mean()) از آرایه‌های موجود در scores محاسبه شده
        # و در دیکشنری res ذخیره می‌شوند.
        #  مهم: وقتی در پارامتر scoring به cross_validate یک دیکشنری می‌دهیم تابع cross_validate به طور خودکار اسم تست را ابتدایشان اضافه میکنه

        for k, v in res.items():
            if k != "Model":
                print(f" {k}: {v:.4f}")
        results.append(res)
        #حالا که دیکشنری res با مقادیر میانگین نهایی (که هر کدام یک عدد تکی هستند) پر شده است، این حلقه for اجرا می‌شود.
        #این حلقه روی آیتم‌های دیکشنری res پیمایش می‌کند و
        # هر معیار (مثل Accuracy، F1 Score و...) را به همراه مقدار میانگین آن (با 4 رقم اعشار) چاپ می‌کند.

    # DataFrame و ذخیره CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "cv_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ CV metrics saved: {csv_path}")

    # نمودار مقایسه‌ای
    plot_metrics_comparison(df, output_dir)
    # اینکه تابع بعدی را داخل این تابع صدا میزنیم:
    # اگر plot_metrics_comparison(df, output_dir) را مستقیماً زیر تعریف تابع plot_metrics_comparison می‌نوشتیم،
    # آن خط بلافاصله پس از تعریف تابع اجرا می‌شد (به محض اینکه فایل پایتون بارگذاری شود)
    # و نه زمانی که شما evaluate_models_cv را فراخوانی می‌کنید. این کار باعث می‌شود:
    #     کد نامنظم شود: توابع بدون هیچ فراخوانی صریح، به محض بارگذاری فایل اجرا شوند.
    #     وابستگی به متغیرهای خارج از scope: df و output_dir در آن زمان هنوز تعریف نشده‌اند و با خطای NameError مواجه می‌شدیم.

    return results

# ──────────────────────────────
# 2)  توابع کمکی رسم نمودارها
# ──────────────────────────────
def plot_metrics_comparison(df, output_dir="outputs"):
    """Bar-chart مقایسه Accuracy / F1 / ROC-AUC بین مدل‌ها را ذخیره می‌کند."""
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))

    # تبدیل ستون هااز حالت "wide" به "long"
    df_melted = df.melt(id_vars="Model",
                        value_vars=["Accuracy", "F1 Score", "ROC AUC"],
                        var_name="Metric",
                        value_name="Score")
    # تابع melt دیتافریم را از حالت پهن (wide) به حالت باریک (long) تبدیل می‌کند تا بتوان با Seaborn نمودار رسم کرد.
    # حالت پهن یا wide یعنی هر متریک، یک ستون جداست چون اطلاعات در ستون‌های مختلف پهن شدن.
    #  اما وقتی تمام متریک ها را در یک ستون ردیف کنیم فرمت Long Format یا باریک را خواهیم داشت.
    # var_name و value_name فقط نام ستون نیستند، بلکه ستون‌هایی هستند که داده‌ها را هم در خود نگه می‌دارند.
    #  پارامتر id_vars:
    # ستون هویتی است و چیزیست که نمی‌خوای تغییر کنه (مثلاً نام مدل‌ها مثل Logistic)
    #
    # - پارامتر value_vars:
    # "ستون‌هایی که می‌خوای باریک کنی"
    # مثلاً Accuracy, F1 Score, ROC AUC
    #
    # - پارامتر  var_name :
    # "اسم ستون جدید برای اسم‌های قدیمی"
    # مثلاً Metric, که توش میاد: Accuracy, F1 Score, ...
    # به عبارت دیگرvar_name نام ستونی است که عنوان ستون‌های value_vars را در خود نگه می‌دارد.
    #
    # - پارامتر value_name:
    # " ستون جدید برای مقادیر".
    # مثلاً Score, که توش میاد: 0.85, 0.88, ...
    # value_name نام ستونی است که مقادیر عددی مربوط به آن عنوان‌ها را در خود نگه می‌دارد.

    sns.barplot(data=df_melted, x="Model", y="Score", hue="Metric")
    # hue="Metric" یعنی به ازای متریک ها رنگ در نظر بگیره
    #  نکته: برتری سیبورن در اینجا نسبت به بارچارت مت پلات لیب:
    # همانطوریکه میبینیم DataFrame را به فرمت "long" یا df_melted تبدیل کردیم.
    # Seaborn به طور طبیعی برای کار با داده‌های "long format" طراحی شده است.

    # با یک خط کد ساده Seaborn به طور خودکار کارهای زیر را انجام میده:
    # 1. گروه‌بندی (بر اساس Model و Metric)،
    # الف.  محور x (Model): Seaborn مقادیر منحصر به فرد ستون "Model" (مثلاً "Logistic Regression", "Decision Tree" و ...) رو شناسایی می‌کنه
    # و برای هر کدوم یک گروه اصلی روی محور X ایجاد می‌کنه.
    # ب. پارامتر hue (Metric): اینجا جادوی گروه‌بندی Seaborn نمایان میشه.
    # hue="Metric" به Seaborn میگه که درون هر گروه اصلیِ Model، زیرگروه‌هایی بر اساس مقادیر ستون "Metric" (Accuracy, F1 Score, ROC AUC) ایجاد کن.
    # 2.محاسبه میانگین (اگر چندین مقدار Score برای یک Model/Metric وجود داشت اما در اینجا ما قبلا میانگیری را انجام دادیم)
    # 3.و رسم میله‌ها را انجام می‌دهد.

    plt.ylim(0, 1)
    # این خط تعیین می‌کند که محور y نمودار از مقدار ۰ تا ۱ باشد.

    plt.title("📊 Metrics Comparison Between Models")

    plt.tight_layout()
    #  این تابع کمک میکند که: جلوگیری از همپوشانی عناصر نمودار.
    # تنظیم فاصله بین زیرنمودارها.
    # تنظیم حاشیه‌های اطراف نمودار (بالا، پایین، چپ، راست).
    # اطمینان از اینکه همه برچسب‌ها، عنوان‌ها و راهنماها در محدوده شکل جای بگیرند.

    path = os.path.join(output_dir, "metrics_comparison.png")
    plt.savefig(path)
    plt.close()
    print(f"🖼 Metrics comparison chart saved: {path}")

def plot_roc_curves(models, X_test, y_test, output_dir="outputs"):
    """منحنی ROC همهٔ مدل‌ها را در یک شکل رسم و ذخیره می‌کند."""
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
        #تابع hasattr() میگه آیا یک شیء دارای یک صفت (attribute) یا متد (method) خاص است یا خیر اگر بود True را برمیگرداند
            y_score = model.predict_proba(X_test)[:, 1]
            #خروجی دو بعدی است بنابراین [:, 1] را قرار دادیم
            #چون دو کلاس داشتیم و اگر کلاسهای بیشتری داشتیم حتی خروجی چند بعدی میشد

        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
            #در اینجا نیازی به انتخاب ستون خاصی نیست، زیرا خروجی یک بعدی است بنابراین یک ستون وجود دارد

        else:
            print(f"⚠️ {name} cannot produce probability scores.")
            continue
        fpr, tpr, _ = roc_curve(y_test, y_score)
        # این تابع سه خروجی را برمی‌گرداند:
        #     fpr: نرخ مثبت کاذب (False Positive Rate)
        #     tpr: نرخ مثبت واقعی (True Positive Rate)
        #     thresholds: آستانه‌ها (thresholds) که در آن‌ها FPR و TPR محاسبه شده‌اند.
        # متغیر آندرلاین (_) در پایتون یک قرارداد (convention) است که به معنای "این متغیر را نادیده بگیر" یا "من به این مقدار نیازی ندارم" است.

        roc_auc = auc(fpr, tpr)
        # عبارت auc مساحت زیر هر نموداری را محاسبه میکند

        # منحنی‌های ROC همه مدل‌ها (چون داخل حلقه for است) روی یک نمودار واحد رسم می‌شوند
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")

    # رسم خط چین  روی نمودار ROC  که از نقطه (0,0) به (1,1) می‌رود.
    # که پس‌زمینه برای مقایسه عملکرد است
    plt.plot([0, 1], [0, 1], "k--")
    # عبارت [0, 1] منظور مختصات است
    # عبارت  k-- یعنی مشکی با استایل __ است
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(output_dir, "roc_curves.png")
    plt.savefig(path)
    plt.close()
    print(f"✅ ROC curves saved: {path}")

