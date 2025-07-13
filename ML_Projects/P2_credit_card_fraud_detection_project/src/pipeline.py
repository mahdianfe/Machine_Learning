# در این فایل از پایپ‌لاین برای پیش‌پردازش و آموزش استفاده می‌شود.

# مربوط به گام 7 و 12
# src/pipeline.py
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def build_logistic_regression_pipeline():
    """Builds a pipeline for Logistic Regression with StandardScaler."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', tol=0.001))
    ])

def build_random_forest_pipeline():
    """Builds a pipeline for RandomForestClassifier with StandardScaler."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier())
    ])

def build_svm_pipeline():
    """Builds a pipeline for SVC with StandardScaler."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(probability=True))
    ])

