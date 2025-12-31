import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
import matplotlib.pyplot as plt

# ========= 1) LOAD DATA =========
df = pd.read_csv("creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

# ========= 2) TRAIN/TEST SPLIT =========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Helper to compute all metrics
def compute_metrics(y_true, y_pred, y_prob):
    return (
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred, zero_division=0),
        recall_score(y_true, y_pred, zero_division=0),
        f1_score(y_true, y_pred, zero_division=0),
        roc_auc_score(y_true, y_prob),
        average_precision_score(y_true, y_prob),
    )

# Helper to print metrics nicely
def print_block(name, acc, prec, rec, f1, roc, auprc):
    print(f"\n==== {name} ====")
    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1 Score :", f1)
    print("ROC-AUC  :", roc)
    print("AUPRC    :", auprc)

# ========= 3) LOGISTIC REGRESSION BEFORE PCA =========
scaler_before = StandardScaler()
X_train_scaled_before = scaler_before.fit_transform(X_train)
X_test_scaled_before = scaler_before.transform(X_test)

log_before = LogisticRegression(max_iter=5000, class_weight="balanced")
log_before.fit(X_train_scaled_before, y_train)

probs_before = np.asarray(log_before.predict_proba(X_test_scaled_before))[:, 1]
pred_before = log_before.predict(X_test_scaled_before)

acc_b, prec_b, rec_b, f1_b, roc_b, auprc_b = compute_metrics(
    y_test, pred_before, probs_before
)

print_block("LOGISTIC REGRESSION BEFORE PCA",
            acc_b, prec_b, rec_b, f1_b, roc_b, auprc_b)

# ========= 4) LOGISTIC REGRESSION AFTER PCA =========
scaler_after = StandardScaler()
X_train_scaled_after = scaler_after.fit_transform(X_train)
X_test_scaled_after = scaler_after.transform(X_test)

pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled_after)
X_test_pca = pca.transform(X_test_scaled_after)

log_after = LogisticRegression(max_iter=5000, class_weight="balanced")
log_after.fit(X_train_pca, y_train)

probs_after = np.asarray(log_after.predict_proba(X_test_pca))[:, 1]
pred_after = log_after.predict(X_test_pca)

acc_a, prec_a, rec_a, f1_a, roc_a, auprc_a = compute_metrics(
    y_test, pred_after, probs_after
)

print_block("LOGISTIC REGRESSION AFTER PCA (95%)",
            acc_a, prec_a, rec_a, f1_a, roc_a, auprc_a)

# ========= 5) BAR PLOT COMPARISON =========
metrics_names = ["ROC-AUC", "AUPRC", "Precision", "Recall", "F1-score", "Accuracy"]

before_scores = [roc_b, auprc_b, prec_b, rec_b, f1_b, acc_b]
after_scores  = [roc_a, auprc_a, prec_a, rec_a, f1_a, acc_a]

x = np.arange(len(metrics_names))
width = 0.35

plt.figure(figsize=(10, 5))
plt.bar(x - width/2, before_scores, width, label="Before PCA")
plt.bar(x + width/2, after_scores,  width, label="After PCA (95%)")

plt.xticks(x, metrics_names, rotation=15)
plt.ylabel("Score")
plt.title("Logistic Regression Performance Before and After PCA")
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()
plt.savefig("logreg_pca_bar_metrics.png", dpi=300)
plt.show()
