
# MACHINEGUARD - PREDICTIVE MAINTENANCE
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier

# 1. CREATE REQUIRED DIRECTORIES
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# 2. LOAD DATASET
df = pd.read_csv("ai4i2020.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())
print("\nDataset information:")
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())

# 3. DATA CLEANING
df.drop_duplicates(inplace=True)
df.drop(columns=["UDI", "Product ID"], inplace=True)
target = "Machine failure"

# 4. FEATURE ENGINEERING
def create_features(data):
    data = data.copy()
    data["Temperature_difference"] = (
        data["Process temperature [K]"]
        - data["Air temperature [K]"]
    )
    data["Mechanical_load_proxy"] = (
        data["Torque [Nm]"]
        * data["Rotational speed [rpm]"]
    )
    return data

# 5. REMOVE TARGET LEAKAGE COLUMNS
leakage_columns = ["TWF", "HDF", "PWF", "OSF", "RNF"]

df.drop(
    columns=leakage_columns,
    errors="ignore",
    inplace=True
)
df = create_features(df)
X = df.drop(columns=[target])
y = df[target]
print("\nFeature columns:")
print(X.columns.tolist())
print("\nTarget distribution:")
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

categorical_features = ["Type"]

numerical_features = [
    col for col in X.columns
    if col not in categorical_features
]
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=(
            (y_train == 0).sum() /
            (y_train == 1).sum()
        ),
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
}

# TRAIN AND COMPARE MODELS

results = []
trained_models = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )
    roc_auc = roc_auc_score(y_test, probabilities)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": roc_auc
    })

    trained_models[name] = pipeline

    print(classification_report(
        y_test, predictions, zero_division=0
    ))


results_df = pd.DataFrame(results)

results_df.sort_values(
    by="F1 Score",
    ascending=False,
    inplace=True
)

print("\nMODEL COMPARISON:")
print(results_df.to_string(index=False))

results_df.to_csv(
    "outputs/model_comparison.csv",
    index=False
)

plt.figure(figsize=(10, 5))

sns.barplot(
    data=results_df,
    x="Model",
    y="F1 Score"
)

plt.title("Model Comparison - F1 Score")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("outputs/model_comparison.png")
plt.close()

# HYPERPARAMETER TUNING
print("\nStarting Random Forest hyperparameter tuning...")
rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

parameter_grid = {
    "classifier__n_estimators": [100, 200, 300, 400],
    "classifier__max_depth": [5, 10, 15, 20, None],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
    "classifier__max_features": ["sqrt", "log2"]
}

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)
random_search = RandomizedSearchCV(
    estimator=rf_pipeline,
    param_distributions=parameter_grid,
    n_iter=10,
    scoring="f1",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=1
)
random_search.fit(X_train, y_train)
best_rf = random_search.best_estimator_
print("\nBest parameters:")
print(random_search.best_params_)
print("\nBest cross-validation F1:")
print(random_search.best_score_)

#  SELECT BEST MODEL

candidates = dict(trained_models)
candidates["Tuned Random Forest"] = best_rf
candidate_scores = {}
for name, model in candidates.items():
    predictions = model.predict(X_test)
    candidate_scores[name] = f1_score(
        y_test,
        predictions,
        zero_division=0
    )
best_name = max(candidate_scores, key=candidate_scores.get)
best_model = candidates[best_name]
print("\nSelected model:", best_name)

# FINAL MODEL EVALUATION

final_predictions = best_model.predict(X_test)
final_probabilities = best_model.predict_proba(X_test)[:, 1]
print("\nFINAL MODEL PERFORMANCE")
print("Accuracy:",
      accuracy_score(y_test, final_predictions))
print("Precision:",
      precision_score(y_test, final_predictions, zero_division=0))
print("Recall:",
      recall_score(y_test, final_predictions, zero_division=0))
print("F1 Score:",
      f1_score(y_test, final_predictions, zero_division=0))
print("ROC-AUC:",
      roc_auc_score(y_test, final_probabilities))
print("\nClassification Report:")
print(classification_report(
    y_test, final_predictions, zero_division=0
))

cm = confusion_matrix(y_test, final_predictions)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Failure", "Failure"],
    yticklabels=["No Failure", "Failure"]
)
plt.title("Confusion Matrix - Best Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png")
plt.close()

print("\nCalculating permutation importance...")

importance = permutation_importance(
    best_model,
    X_test,
    y_test,
    scoring="f1",
    n_repeats=5,
    random_state=42,
    n_jobs=-1
)

importance_df = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": importance.importances_mean
}).sort_values(
    by="Importance",
    ascending=False
)

print(importance_df)

importance_df.to_csv(
    "outputs/feature_importance.csv",
    index=False
)

plt.figure(figsize=(9, 6))

sns.barplot(
    data=importance_df,
    x="Importance",
    y="Feature"
)

plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png")
plt.close()

# SAVE BEST MODEl
joblib.dump(
    best_model,
    "models/best_model.pkl"
)
# Save model information
metadata = {
    "model_name": best_name,
    "features": X.columns.tolist(),
    "accuracy": float(
        accuracy_score(y_test, final_predictions)
    ),
    "precision": float(
        precision_score(y_test, final_predictions, zero_division=0)
    ),
    "recall": float(
        recall_score(y_test, final_predictions, zero_division=0)
    ),
    "f1_score": float(
        f1_score(y_test, final_predictions, zero_division=0)
    ),
    "roc_auc": float(
        roc_auc_score(y_test, final_probabilities)
    )
}

with open("models/model_metadata.json", "w") as file:
    json.dump(metadata, file, indent=4)

print("\n===================================")
print("MODEL TRAINING COMPLETED!")
print("Best model:", best_name)
print("Model saved successfully!")
print("===================================")
