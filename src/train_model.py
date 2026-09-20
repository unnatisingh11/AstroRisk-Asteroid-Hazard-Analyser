import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score
)


# -----------------------------
# 1. Load processed dataset
# -----------------------------

DATA_PATH = "data/processed/clean_neo_data.csv"
MODEL_PATH = "models/astrorisk_rf.pkl"

df = pd.read_csv(DATA_PATH)

print("Dataset loaded.")
print("Shape:", df.shape)


# -----------------------------
# 2. Select features and target
# -----------------------------

# IMPORTANT:
# We intentionally DO NOT use:
# moid -> directly related to PHA definition
# H -> directly related to PHA definition
# neo -> all records are NEOs
# diameter/albedo -> too much missing data originally
# identifiers -> not useful for prediction
# class_* -> derived orbital classification; excluded from baseline

features = [
    "e",
    "a",
    "q",
    "i",
    "per",
    "condition_code",
    "data_arc",
    "n_obs_used"
]

target = "pha"

X = df[features]
y = df[target]


print("\nFeatures used:")
for feature in features:
    print("-", feature)

print("\nTarget:", target)


# -----------------------------
# 3. Check target distribution
# -----------------------------

print("\nTarget distribution:")
print(y.value_counts())

print("\nTarget percentage:")
print(y.value_counts(normalize=True) * 100)


# -----------------------------
# 4. Train-test split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# -----------------------------
# 5. Create Random Forest
# -----------------------------

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)


# -----------------------------
# 6. Train model
# -----------------------------

print("\nTraining Random Forest...")

model.fit(X_train, y_train)

print("Training completed.")


# -----------------------------
# 7. Make predictions
# -----------------------------

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]


# -----------------------------
# 8. Evaluate model
# -----------------------------

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_probability)

print("\n" + "=" * 50)
print("MODEL RESULTS")
print("=" * 50)

print("\nAccuracy:", round(accuracy, 4))
print("ROC-AUC:", round(roc_auc, 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# -----------------------------
# 9. Feature importance
# -----------------------------

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    by="importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance)


# -----------------------------
# 10. Save model
# -----------------------------

model_data = {
    "model": model,
    "features": features
}

joblib.dump(model_data, MODEL_PATH)

print("\nModel saved to:")
print(MODEL_PATH)