# src/evaluate_nb_model.py

import os
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_nb_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    # چاپ در کنسول
    print("Naive Bayes Classification Report:")
    print(report)
    print("Naive Bayes Confusion Matrix:")
    print(matrix)

    # ذخیره در فایل
    output_path = os.path.join("evaluation_outputs", "nb_report.txt")
    with open(output_path, 'w') as f:
        f.write("Naive Bayes Classification Report:\n")
        f.write(report + "\n\n")
        f.write("Naive Bayes Confusion Matrix:\n")
        f.write(str(matrix))

"""
برای خروجی این فایل باید از فایل evaluate_models.py خروجی گرفته شود  
"""