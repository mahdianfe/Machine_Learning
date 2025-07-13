# src/evaluate_lr_model.py




import os
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_lr_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    print("Logistic Regression Classification Report:")
    print(report)
    print("Logistic Regression Confusion Matrix:")
    print(matrix)

    output_path = os.path.join("evaluation_outputs", "lr_report.txt")
    with open(output_path, 'w') as f:
        f.write("Logistic Regression Classification Report:\n")
        f.write(report + "\n\n")
        f.write("Logistic Regression Confusion Matrix:\n")
        f.write(str(matrix))


"""
برای خروجی این فایل باید از فایل evaluate_models.py خروجی گرفته شود  
"""