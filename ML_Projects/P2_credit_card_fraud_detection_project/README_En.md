# 🛡️ Credit Card Fraud Detection Project – P2_Credit_Card_Fraud_Detection

This project implements a machine learning pipeline to detect fraudulent credit card transactions. The goal is to build a robust fraud detection system that can accurately flag suspicious activity in financial transaction data.

---

## 🧠 Technical Summary

- Uses Logistic Regression, with the possibility of extension to more advanced models (e.g., Random Forest, XGBoost).
- Data validation (missing values, class labels, negative amounts, time ranges).
- Data preprocessing (outlier clipping, normalization, NA removal).
- Feature engineering and scaling.
- Train/test split and persistent storage.
- Model training and evaluation using comprehensive metrics.
- ROC curve plotting for all models.
- Hyperparameter tuning via GridSearchCV.
- Anomaly detection using Isolation Forest.
- Final storage of trained models and performance metrics.

---

## 🗂 Project Structure

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

## 🚀 Pipeline Steps

### 1. 📥 Data Loading
From `data/raw/creditcard.csv` using `pandas.read_csv`.

### 2. ✅ Data Validation
Checks for:
- No NaN values
- Valid `Class` labels (only 0 and 1)
- Non-negative `Amount`
- Reasonable values in `Time` (e.g., less than 48 hours)

### 3. 🧼 Preprocessing
- Clipping outliers in `Amount`
- Normalizing `Amount` and `Time` using `StandardScaler`
- Dropping duplicate rows

### 4. 🔬 Feature Engineering
- Scaling all features
- Separating `X` and `y`

### 5. ✂️ Train/Test Split & Save
- Using `train_test_split`
- Saving processed datasets

### 6. 🧱 Pipeline Creation
- Pipeline with `StandardScaler` and model

### 7. 🏋️ Model Training
Logistic Regression as initial model.

### 8. 📊 Model Evaluation with Cross Validation
Metrics:
- Accuracy
- F1 Score
- Precision
- Recall
- ROC AUC

### 9. 📈 ROC Curve Plotting
Using `matplotlib` for visual comparison of models.

### 10. ⚙️ Hyperparameter Tuning
- Via `GridSearchCV` on `RandomForestClassifier`

### 11. 🚨 Anomaly Detection
Using `IsolationForest` on the test set.

### 12. 💾 Save Outputs
Models, metrics, ROC plots, and comparison charts are saved.

---

## ⚙️ How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Full Pipeline
```bash
python src/main.py
```

---

## 🔍 Dataset Source

- [Credit Card Fraud Detection Dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## 📚 Notebooks

- `data_analysis.ipynb`: Exploratory Data Analysis (EDA)
- `model_training_evaluation.ipynb`: Model building and evaluation

---

## 📜 License

This project is licensed under the MIT License.

---

## 🤝 Contributing

Pull requests are welcome!  
To contribute:

1. Fork the repo 🍴  
2. Create a new branch 🛠  
3. Submit a pull request ✅  

---

## ☕ Support the Developer

If this project helped you, consider buying me a coffee:

[Buy Me a Coffee](https://www.coffeebede.com/mahdianfe)

---

**Made with 💻, love, and lots of ☕ to fight fraud!**
