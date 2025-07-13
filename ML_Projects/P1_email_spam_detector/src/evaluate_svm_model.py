#  src/evaluate_svm_model.py
# # ارزیابی مدل اس وی ام
#
# from sklearn.metrics import classification_report, confusion_matrix
#
# def evaluate_svm_model(model, X_test, y_test):
#     y_pred = model.predict(X_test)
#
#     # گزارش طبقه‌بندی
#     print("SVM Classification Report:")
#     print(classification_report(y_test, y_pred))
#
#     # ماتریس سردرگمی
#     print("SVM Confusion Matrix:")
#     print(confusion_matrix(y_test, y_pred))
#########
import os
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_svm_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    # چاپ در کنسول
    print("SVM Classification Report:")
    print(report)
    print("SVM Confusion Matrix:")
    print(matrix)

    # ذخیره در فایل
    output_path = os.path.join("evaluation_outputs", "svm_report.txt")
    with open(output_path, 'w') as f:
        f.write("SVM Classification Report:\n")
        f.write(report + "\n\n")
        f.write("SVM Confusion Matrix:\n")
        f.write(str(matrix))

"""
برای خروجی این فایل باید از فایل evaluate_models.py خروجی گرفته شود  
"""