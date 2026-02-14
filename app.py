import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')  # ✅ MUST be before pyplot import - fixes Streamlit Cloud error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Classification - ML Models",
    page_icon="🫀",
    layout="wide"
)

# ─────────────────────────────────────────────
# Title
# ─────────────────────────────────────────────
st.title("🫀 Heart Disease Classification - ML Model Comparison")
st.markdown("""
This application demonstrates **6 machine learning classification models**
trained on the Heart Disease dataset with full evaluation metrics.
""")

# ─────────────────────────────────────────────
# Sidebar - Model Selection & File Upload
# ─────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")

model_options = {
    'Logistic Regression':         'logistic_regression.pkl',
    'Decision Tree':               'decision_tree.pkl',
    'K-Nearest Neighbor':          'k_nearest_neighbor.pkl',
    'Naive Bayes':                 'naive_bayes.pkl',
    'Random Forest':               'random_forest.pkl',
    'XGBoost (Gradient Boosting)': 'xgboost.pkl'
}

selected_model = st.sidebar.selectbox(
    "Select Model",
    options=list(model_options.keys())
)

st.sidebar.markdown("---")
st.sidebar.header("📂 Upload Test Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file (use test_data.csv)",
    type=['csv'],
    help="Upload test_data.csv provided with this project"
)

# ─────────────────────────────────────────────
# Helper: Load model safely
# ─────────────────────────────────────────────
@st.cache_resource
def load_model(model_name):
    model_file = model_options[model_name]
    if not os.path.exists(model_file):
        return None
    with open(model_file, 'rb') as f:
        return pickle.load(f)

# ─────────────────────────────────────────────
# Helper: Load results safely
# ─────────────────────────────────────────────
@st.cache_data
def load_results():
    if not os.path.exists('model_results.csv'):
        return None
    return pd.read_csv('model_results.csv')

# ─────────────────────────────────────────────
# Section 1: Model Comparison Table
# ─────────────────────────────────────────────
st.header("📊 Model Performance Metrics")

results_df = load_results()

if results_df is not None:
    st.dataframe(
        results_df.style.highlight_max(
            axis=0,
            subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'],
            color='lightgreen'
        ).format({
            'Accuracy': '{:.4f}', 'AUC': '{:.4f}', 'Precision': '{:.4f}',
            'Recall':   '{:.4f}', 'F1':  '{:.4f}', 'MCC':       '{:.4f}'
        }),
        use_container_width=True
    )

    # Bar charts for all metrics
    st.subheader("📈 Visual Comparison Across All Models")
    metrics_list = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    for idx, metric in enumerate(metrics_list):
        ax = axes[idx // 3, idx % 3]
        colors = ['green' if v == results_df[metric].max() else 'steelblue'
                  for v in results_df[metric]]
        bars = ax.bar(results_df['Model'], results_df[metric], color=colors)
        ax.set_title(f'{metric}', fontsize=13, fontweight='bold')
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.grid(axis='y', alpha=0.3)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f'{bar.get_height():.3f}',
                    ha='center', va='bottom', fontsize=7)

    plt.suptitle('All Models - Metric Comparison (Green = Best)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
else:
    st.error("model_results.csv not found. Make sure it is uploaded to GitHub.")

# ─────────────────────────────────────────────
# Section 2: Dataset Info
# ─────────────────────────────────────────────
with st.expander("📝 Dataset Information"):
    st.markdown("""
    **Dataset:** Heart Disease Classification (UCI Repository)

    | Property | Value |
    |---|---|
    | Total Instances | 600 |
    | Features | 13 |
    | Target | Binary (0: No Disease, 1: Disease) |
    | Train Split | 480 (80%) |
    | Test Split | 120 (20%) |

    **Features:** age, sex, cp, trestbps, chol, fbs, restecg,
    thalach, exang, oldpeak, slope, ca, thal
    """)

# ─────────────────────────────────────────────
# Section 3: Predictions on Uploaded File
# ─────────────────────────────────────────────
st.markdown("---")
st.header("🔮 Model Predictions")

if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)
    st.success(f"✅ File uploaded — Shape: {test_data.shape}")

    with st.expander("👁️ Preview Uploaded Data"):
        st.dataframe(test_data.head(10))

    # Separate features and labels
    if 'target' in test_data.columns:
        X_test     = test_data.drop('target', axis=1)
        y_test     = test_data['target']
        has_labels = True
    else:
        X_test     = test_data
        has_labels = False

    # Load chosen model
    model = load_model(selected_model)

    if model is None:
        st.error(f"Model file not found: {model_options[selected_model]}. "
                 "Make sure all .pkl files are uploaded to GitHub.")
    else:
        y_pred       = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        col1, col2 = st.columns(2)

        # Left: predictions table
        with col1:
            st.subheader(f"Predictions — {selected_model}")
            pred_df = pd.DataFrame({
                'Prediction': ['Disease' if p == 1 else 'No Disease' for p in y_pred],
                'Probability': np.round(y_pred_proba, 4)
            })
            st.dataframe(pred_df, use_container_width=True)

            st.metric("Total Samples",           len(y_pred))
            st.metric("Predicted Disease (1)",    int(sum(y_pred)))
            st.metric("Predicted No Disease (0)", int(len(y_pred) - sum(y_pred)))

        # Right: metrics + confusion matrix
        with col2:
            if has_labels:
                st.subheader("📈 Evaluation Metrics")

                accuracy  = accuracy_score(y_test, y_pred)
                auc       = roc_auc_score(y_test, y_pred_proba)
                precision = precision_score(y_test, y_pred)
                recall    = recall_score(y_test, y_pred)
                f1        = f1_score(y_test, y_pred)
                mcc       = matthews_corrcoef(y_test, y_pred)

                m_df = pd.DataFrame({
                    'Metric': ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC'],
                    'Value':  [accuracy, auc, precision, recall, f1, mcc]
                })
                st.dataframe(m_df.style.format({'Value': '{:.4f}'}), use_container_width=True)

                # Confusion Matrix
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig2, ax2 = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                            xticklabels=['No Disease', 'Disease'],
                            yticklabels=['No Disease', 'Disease'])
                ax2.set_ylabel('Actual')
                ax2.set_xlabel('Predicted')
                ax2.set_title(f'Confusion Matrix — {selected_model}')
                st.pyplot(fig2)
                plt.close()

                # Classification Report
                st.subheader("Classification Report")
                report = classification_report(
                    y_test, y_pred,
                    target_names=['No Disease', 'Disease'],
                    output_dict=True
                )
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.format('{:.4f}'), use_container_width=True)

            else:
                st.info("Upload a file with a 'target' column to see full evaluation metrics.")

else:
    st.info("👆 Upload **test_data.csv** from the sidebar to see predictions and metrics.")
    st.markdown("""
    **Expected CSV columns:**
    `age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target`
    """)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:grey;'>"
    "ML Assignment 2 — Heart Disease Classification | "
    "Logistic Regression · Decision Tree · KNN · Naive Bayes · Random Forest · XGBoost"
    "</div>",
    unsafe_allow_html=True
)
