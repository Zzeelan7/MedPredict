import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import eli5
from eli5.sklearn import PermutationImportance
import joblib

# Load data
df = pd.read_csv("HeartDiseaseTrain-Test.csv")

st.title("Disease Prediction - Heart Disease")
st.subheader("Data Preview")
st.dataframe(df.head())

# Correlation heatmap
st.subheader("Correlation Heatmap")
corr = df.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Features & target
features = [
    "cholestoral", "Max_heart_rate",
    "age"
]
X = df[features].dropna()
y = df["target"].loc[X.index]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SMOTE balancing
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Model choice
model_choice = st.selectbox(
    "Choose a model", 
    ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"]
)

# Hyperparameter tuning
if model_choice == "Logistic Regression":
    model = LogisticRegression(class_weight='balanced', max_iter=2000)
    params = {"C": [0.01, 0.1, 1, 10]}

elif model_choice == "Random Forest":
    model = RandomForestClassifier(class_weight='balanced', random_state=42)
    params = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5, 10]
    }

elif model_choice == "XGBoost":
    scale_pos_weight = (len(y_train_resampled) - sum(y_train_resampled)) / sum(y_train_resampled)
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight)
    params = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2]
    }

elif model_choice == "LightGBM":
    model = LGBMClassifier(class_weight='balanced')
    params = {
        "n_estimators": [100, 200],
        "max_depth": [-1, 5, 10],
        "learning_rate": [0.01, 0.1, 0.2]
    }

# Grid search
st.write("Tuning hyperparameters, please wait...")
grid = GridSearchCV(model, params, cv=3, scoring='accuracy', n_jobs=-1)
grid.fit(X_train_resampled, y_train_resampled)
best_model = grid.best_estimator_

# Predictions
y_pred = best_model.predict(X_test_scaled)
y_train_pred = best_model.predict(X_train_scaled)

# Performance metrics
st.subheader("Model Performance")
st.write("Best Parameters:", grid.best_params_)
st.write("Train Accuracy:", accuracy_score(y_train, y_train_pred))
st.write("Test Accuracy:", accuracy_score(y_test, y_pred))
st.write("Recall (Positive class):", recall_score(y_test, y_pred))
st.text("Classification Report:\n" + classification_report(y_test, y_pred))

# Cross-validation score
cv_scores = cross_val_score(best_model, X, y, cv=5)
st.write("Cross-Validation Scores:", cv_scores)
st.write("Mean CV Accuracy:", np.mean(cv_scores))

# Overfitting warning
if accuracy_score(y_train, y_train_pred) - np.mean(cv_scores) > 0.15:
    st.error("⚠️ Possible Overfitting: Large gap between train accuracy and CV accuracy.")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
st.pyplot(fig_cm)

# Feature importance for tree-based models
if model_choice in ["Random Forest", "XGBoost", "LightGBM"]:
    perm = PermutationImportance(best_model, random_state=42).fit(X_test_scaled, y_test)
    st.subheader("Feature Importance")
    st.write(eli5.format_as_dataframe(eli5.explain_weights(perm, feature_names=features)))

# Save model
joblib.dump(best_model, f"{model_choice}_heart_disease_model.pkl")
st.success(f"Model saved as {model_choice}_heart_disease_model.pkl")
