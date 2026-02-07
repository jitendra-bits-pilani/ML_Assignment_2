
# Diabetes Classification ML Assignment

## Problem Statement
Predict early-stage diabetes risk (**Positive/Negative**) using patient symptoms and demographics.  
This project demonstrates a complete ML pipeline: preprocessing, training six classification models, evaluating them with multiple metrics, and deploying an interactive Streamlit app — as per BITS Assignment 2 requirements.

---

## Dataset Description
- **Source**: UCI ML Repository – Early Stage Diabetes Risk Prediction Dataset  
- **File**: `diabetes_data_upload.csv`  
- **Size**: 520 instances, 17 attributes  
- **Features (16)**: Age, Gender, Polyuria, Polydipsia, sudden weight loss, weakness, Polyphagia, Genital thrush, visual blurring, Itching, Irritability, delayed healing, partial paresis, muscle stiffness, Alopecia, Obesity  
- **Target**: `class` (Positive = 1, Negative = 0 diabetes risk)  

Meets assignment requirements: >500 instances, >12 features, binary classification.

---

## Models Implemented
1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbor (kNN)  
4. Naive Bayes (GaussianNB)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)  

---

## Model Comparison Table

| Model              | Accuracy | AUC   | Precision | Recall | F1    | MCC   |
|--------------------|----------|-------|-----------|--------|-------|-------|
| Logistic Regression| 0.9231   | 0.9774| 0.9315    | 0.9577 | 0.9444| 0.8204|
| Decision Tree      | 0.9615   | 0.9718| 1.0000    | 0.9437 | 0.9710| 0.9174|
| kNN                | 0.8942   | 0.9774| 0.9545    | 0.8873 | 0.9197| 0.7698|
| Naive Bayes        | 0.9135   | 0.9607| 0.9306    | 0.9437 | 0.9371| 0.7988|
| Random Forest      | 0.9904   | 1.0000| 1.0000    | 0.9859 | 0.9929| 0.9782|
| XGBoost            | 0.9712   | 1.0000| 1.0000    | 0.9577 | 0.9784| 0.9370|

---

## Model Performance Observations

| Model              | Observation |
|--------------------|-------------|
| Logistic Regression| Reliable baseline (92% accuracy), excellent AUC. Balanced performance suitable for medical diagnosis. |
| Decision Tree      | Perfect precision (no false positives), interpretable. Good for clinical explanation but potential overfitting. |
| kNN                | Solid performance but distance-based → sensitive to scaling. Lowest MCC indicates imbalance handling issues. |
| Naive Bayes        | Fastest inference, strong recall (catches most diabetics). Independence assumption limits top performance. |
| Random Forest      | Best overall — near-perfect scores across all metrics. Robust ensemble handles noise/overfitting best. |
| XGBoost            | Excellent performer, perfect precision/AUC. Slightly conservative recall but highest deployment reliability. |

**Key Insight**: Ensemble methods (Random Forest, XGBoost) dominate due to dataset complexity.  
Random Forest wins for production deployment.

---

## Repository Structure
```
project/
├── app.py                    # Streamlit web app
├── train_models.py           # Model training script
├── requirements.txt          # Dependencies
├── diabetes_data_upload.csv  # Dataset
├── model/                    # Trained models + preprocessing
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── feature_scaler.pkl
│   ├── label_encoders.pkl
│   └── results.csv
└── README.md
```

---

## Deployment
- **Live App**: https://mlassignment2-jhezzjrsfvbugtkmzcnkuq.streamlit.app/ 
- **Features**:
  - CSV upload (test data only for free tier)  
  - Model selection dropdown (6 models)  
  - Full metrics display (classification report + confusion matrix)  
  - Confusion matrix visualization  
  - Individual patient predictions  

---

## 📦 Requirements
```
streamlit
scikit-learn
xgboost
pandas
numpy
matplotlib
seaborn
joblib
```

---

## ▶️ How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train models
```bash
python train_models.py
```

### 3. Run Streamlit app
```bash
python -m streamlit run app.py
```

### 4. Deploy
- Push repo to GitHub  
- Deploy via [Streamlit Cloud](https://streamlit.io/cloud)  
- live app link : https://mlassignment2-jhezzjrsfvbugtkmzcnkuq.streamlit.app/ 

---
```