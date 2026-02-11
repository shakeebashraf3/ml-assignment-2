import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef, 
                             confusion_matrix, classification_report)
import pickle
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare the dataset"""
    df = pd.read_csv('heart_disease.csv')
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save test data for Streamlit app
    test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    test_df['target'] = y_test.values
    test_df.to_csv('test_data.csv', index=False)
    
    # Save scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train all models and evaluate them"""
    
    # Note: Using GradientBoostingClassifier as XGBoost equivalent (both are gradient boosting methods)
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'K-Nearest Neighbor': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': GradientBoostingClassifier(n_estimators=100, random_state=42, learning_rate=0.1)
    }
    
    results = []
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'Model': name,
            'Accuracy': round(accuracy_score(y_test, y_pred), 4),
            'AUC': round(roc_auc_score(y_test, y_pred_proba), 4),
            'Precision': round(precision_score(y_test, y_pred), 4),
            'Recall': round(recall_score(y_test, y_pred), 4),
            'F1': round(f1_score(y_test, y_pred), 4),
            'MCC': round(matthews_corrcoef(y_test, y_pred), 4)
        }
        
        results.append(metrics)
        
        # Save model
        model_filename = f"{name.lower().replace(' ', '_').replace('-', '_')}.pkl"
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"Metrics for {name}:")
        for key, value in metrics.items():
            if key != 'Model':
                print(f"  {key}: {value}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv('model_results.csv', index=False)
    
    return results_df

if __name__ == "__main__":
    print("Loading and preparing data...")
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    
    print(f"\nTraining set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    print("\n" + "="*60)
    print("Training and Evaluating Models")
    print("="*60)
    
    results_df = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)
    print(results_df.to_string(index=False))
    
    print("\nAll models trained and saved successfully!")
    print("Files created:")
    print("  - model_results.csv")
    print("  - test_data.csv")
    print("  - scaler.pkl")
    print("  - Individual model .pkl files")
