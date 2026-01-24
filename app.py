import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    matthews_corrcoef
)
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Diabetes Classification Models", layout="wide")
st.title("🩺 Diabetes Classification Models")

# Upload test dataset
uploaded_file = st.file_uploader("Upload CSV file", type="csv")
if uploaded_file:
    test_data = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(test_data.head())

    # Encode categorical features
    encoded_data = test_data.copy()
    for col in encoded_data.columns:
        if encoded_data[col].dtype == 'object':
            encoded_data[col] = LabelEncoder().fit_transform(encoded_data[col])

    # Separate features and target
    if "class" not in encoded_data.columns:
        st.error("Uploaded data must contain a 'class' column for prediction.")
    else:
        X_test = encoded_data.drop("class", axis=1)
        y_test = encoded_data["class"]

        # Scale features
        scaler = StandardScaler()
        X_test_scaled = scaler.fit_transform(X_test)

        # Model selection
        model_choice = st.selectbox(
            "Select a model to run",
            ["Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Random Forest", "XGBoost"]
        )

        if st.button("Run Model"):
            try:
                with open(f"model/{model_choice.replace(' ', '_').lower()}.pkl", "rb") as f:
                    model = pickle.load(f)

                y_pred = model.predict(X_test_scaled)

                # Map labels for readability
                label_map = {0: "Negative", 1: "Positive"}
                y_test_named = pd.Series(y_test).map(label_map)
                y_pred_named = pd.Series(y_pred).map(label_map)

                # Classification report
                st.subheader("📊 Classification Report")
                report = classification_report(y_test_named, y_pred_named, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.format("{:.2f}"))

                # Confusion matrix (dynamic)
                st.subheader("🧮 Confusion Matrix")
                all_labels = sorted(set(y_test_named) | set(y_pred_named))
                cm = confusion_matrix(y_test_named, y_pred_named, labels=all_labels)
                cm_df = pd.DataFrame(
                    cm,
                    index=[f"Actual {label}" for label in all_labels],
                    columns=[f"Predicted {label}" for label in all_labels]
                )
                fig, ax = plt.subplots()
                sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax)
                st.pyplot(fig)

                # Additional metrics
                st.subheader("📌 Additional Metrics")
                accuracy = accuracy_score(y_test, y_pred)
                try:
                    auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
                except Exception:
                    auc = "N/A"
                mcc = matthews_corrcoef(y_test, y_pred)

                st.markdown(f"""
                - **Accuracy**: `{accuracy:.4f}`  
                - **AUC Score**: `{auc if isinstance(auc, str) else f"{auc:.4f}"}`  
                - **Matthews Correlation Coefficient (MCC)**: `{mcc:.4f}`
                """)

            except Exception as e:
                st.error(f"Error loading model or running prediction: {e}")