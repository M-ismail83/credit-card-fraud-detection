import numpy as np
import pandas as pd
import time  # ✅ NEW: for timing

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib.pyplot as plt

# ============= TOTAL SCRIPT TIME START =============
overall_start = time.time()

# ============= 1) LOAD DATA =============
df = pd.read_csv("creditcard.csv")   # تأكدي من الاسم

X = df.drop("Class", axis=1)
y = df["Class"]

# ============= 2) TRAIN / TEST SPLIT =============
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

def compute_metrics(y_true, y_pred):
    """Return Accuracy, Precision, Recall, F1."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return acc, prec, rec, f1

def print_metrics_block(title, acc, prec, rec, f1):
    print(f"\n==== {title} ====")
    print(f"Accuracy : {acc}")
    print(f"Precision: {prec}")
    print(f"Recall   : {rec}")
    print(f"F1 Score : {f1}")

# ============= 3) KNN BEFORE PCA =============
# KNN is distance-based → needs scaling
scaler_before = StandardScaler()
X_train_scaled = scaler_before.fit_transform(X_train)
X_test_scaled = scaler_before.transform(X_test)

knn_before = KNeighborsClassifier(
    n_neighbors=5,
    weights="distance",   # يعطي وزن أكبر للجيران الأقرب
    metric="minkowski",   # default (Euclidean)
    n_jobs=-1
)

# ⏱️ Time KNN BEFORE PCA (fit + predict)
start_before = time.time()
knn_before.fit(X_train_scaled, y_train)
y_pred_before = knn_before.predict(X_test_scaled)
end_before = time.time()

acc_b, prec_b, rec_b, f1_b = compute_metrics(y_test, y_pred_before)

print_metrics_block("KNN BEFORE PCA",
                    acc_b, prec_b, rec_b, f1_b)
print(f"Processing time (KNN BEFORE PCA): {end_before - start_before:.2f} seconds")

# ============= 4) KNN AFTER PCA =============
scaler_after = StandardScaler()
X_train_scaled2 = scaler_after.fit_transform(X_train)
X_test_scaled2 = scaler_after.transform(X_test)

# PCA: احتفظ مثلاً بـ 95% من التباين
pca = PCA(n_components=0.95)

# ⏱️ Time PCA step
start_pca = time.time()
X_train_pca = pca.fit_transform(X_train_scaled2)
X_test_pca = pca.transform(X_test_scaled2)
end_pca = time.time()

print(f"\nOriginal features: {X_train.shape[1]}")
print(f"PCA features     : {X_train_pca.shape[1]}")
print(f"PCA processing time: {end_pca - start_pca:.2f} seconds")

knn_after = KNeighborsClassifier(
    n_neighbors=5,
    weights="distance",
    metric="minkowski",
    n_jobs=-1
)

# ⏱️ Time KNN AFTER PCA (fit + predict)
start_after = time.time()
knn_after.fit(X_train_pca, y_train)
y_pred_after = knn_after.predict(X_test_pca)
end_after = time.time()

acc_a, prec_a, rec_a, f1_a = compute_metrics(y_test, y_pred_after)

print_metrics_block("KNN AFTER PCA",
                    acc_a, prec_a, rec_a, f1_a)
print(f"Processing time (KNN AFTER PCA): {end_after - start_after:.2f} seconds")

# ============= 5) BAR PLOT: BEFORE vs AFTER PCA =============
metrics_names = ["Accuracy", "Precision", "Recall", "F1-score"]

before_scores = [acc_b, prec_b, rec_b, f1_b]
after_scores  = [acc_a, prec_a, rec_a, f1_a]

x = np.arange(len(metrics_names))
width = 0.35

plt.figure(figsize=(10, 5))
plt.bar(x - width/2, before_scores, width, label="KNN (Before PCA)")
plt.bar(x + width/2, after_scores,  width, label="KNN (After PCA)")

plt.xticks(x, metrics_names, rotation=15)
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.title("KNN Performance Before and After PCA")
plt.legend()
plt.tight_layout()
plt.savefig("knn_pca_bar_metrics.png", dpi=300)
plt.show()

# ============= TOTAL SCRIPT TIME END =============
overall_end = time.time()
print(f"\nTOTAL script processing time: {overall_end - overall_start:.2f} seconds")
