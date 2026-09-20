import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)


# -----------------------------
# 1. Load dataset and model
# -----------------------------

DATA_PATH = "data/processed/clean_neo_data.csv"
MODEL_PATH = "models/astrorisk_rf.pkl"

df = pd.read_csv(DATA_PATH)

model_data = joblib.load(MODEL_PATH)

model = model_data["model"]
features = model_data["features"]


# -----------------------------
# 2. Prepare data
# -----------------------------

X = df[features]
y = df["pha"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]


# -----------------------------
# 3. Confusion Matrix
# -----------------------------

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Non-PHA", "PHA"],
    yticklabels=["Non-PHA", "PHA"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("AstroRisk Confusion Matrix")

plt.tight_layout()
plt.savefig("models/confusion_matrix.png", dpi=300)
plt.show()


# -----------------------------
# 4. Feature Importance
# -----------------------------

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(8, 5))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("AstroRisk Feature Importance")

plt.tight_layout()
plt.savefig("models/feature_importance.png", dpi=300)
plt.show()


# -----------------------------
# 5. ROC Curve
# -----------------------------

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

auc_score = roc_auc_score(
    y_test,
    y_probability
)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"Random Forest (AUC = {auc_score:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("AstroRisk ROC Curve")

plt.legend()

plt.tight_layout()
plt.savefig("models/roc_curve.png", dpi=300)
plt.show()


print("\nEvaluation complete.")

print("\nGenerated files:")

print("1. models/confusion_matrix.png")
print("2. models/feature_importance.png")
print("3. models/roc_curve.png")