# pca_svm_improvement.py

import pandas as pd
import numpy as np
import time  # ✅ for processing time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
import matplotlib.pyplot as plt

# -----------------------------------------
# 0. Total script time start
# -----------------------------------------
overall_start = time.time()

# -----------------------------------------
# 1. Load the dataset
# -----------------------------------------
df = pd.read_csv("creditcard.csv")

# Split into features (X) and labels (y)
X = df.drop("Class", axis=1)
y = df["Class"]

# -----------------------------------------
# 2. Train-test split (keeps fraud ratio same)
# -----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# -----------------------------------------
# 3. Scale data (required for SVM)
# -----------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------
# 4. SVM WITHOUT PCA (baseline)
# -----------------------------------------
svm_full = SVC(
    kernel="linear",
    probability=True,
    random_state=42,
    max_iter=5000,
    tol=1e-5
)

start_full = time.time()  # ⏱️ start
svm_full.fit(X_train_scaled, y_train)
y_pred_full = svm_full.predict(X_test_scaled)
end_full = time.time()    # ⏱️ end

# Metrics for SVM with all features
accuracy_full = accuracy_score(y_test, y_pred_full)
precision_full = precision_score(y_test, y_pred_full, zero_division=0)
recall_full = recall_score(y_test, y_pred_full, zero_division=0)
f1_full = f1_score(y_test, y_pred_full, zero_division=0)

print("\n===== SVM Model 1: Using ALL Features =====")
print("Accuracy :", accuracy_full)
print("Precision:", precision_full)
print("Recall   :", recall_full)
print("F1       :", f1_full)
print(f"Processing time (SVM without PCA): {end_full - start_full:.2f} seconds")

# -----------------------------------------
# 5. Apply PCA (keep 95% of variance)
# -----------------------------------------
pca = PCA(n_components=0.95)

start_pca = time.time()  # ⏱️ PCA time
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
end_pca = time.time()

print("\nOriginal number of features:", X_train_scaled.shape[1])
print("Number of PCA features:", X_train_pca.shape[1])
print(f"PCA processing time: {end_pca - start_pca:.2f} seconds")

# -----------------------------------------
# 6. SVM WITH PCA
# -----------------------------------------
svm_pca = SVC(
    kernel="linear",
    probability=True,
    random_state=42,
    max_iter=5000,
    tol=1e-5
)

start_pca_svm = time.time()  # ⏱️ SVM with PCA time
svm_pca.fit(X_train_pca, y_train)
y_pred_pca = svm_pca.predict(X_test_pca)
end_pca_svm = time.time()

# Metrics after PCA
accuracy_pca = accuracy_score(y_test, y_pred_pca)
precision_pca = precision_score(y_test, y_pred_pca, zero_division=0)
recall_pca = recall_score(y_test, y_pred_pca, zero_division=0)
f1_pca = f1_score(y_test, y_pred_pca, zero_division=0)

print("\n===== SVM Model 2: Using PCA (95% variance) =====")
print("Accuracy :", accuracy_pca)
print("Precision:", precision_pca)
print("Recall   :", recall_pca)
print("F1       :", f1_pca)
print(f"Processing time (SVM with PCA): {end_pca_svm - start_pca_svm:.2f} seconds")

# -----------------------------------------
# 7. Comparison summary
# -----------------------------------------
print("\n===== COMPARISON SUMMARY (SVM) =====")
print("SVM (All Features) → "
      f"Acc: {accuracy_full:.4f}, Precision: {precision_full:.4f}, "
      f"Recall: {recall_full:.4f}, F1: {f1_full:.4f}, "
      f"Time: {end_full - start_full:.2f}s")

print("SVM (PCA Features) → "
      f"Acc: {accuracy_pca:.4f}, Precision: {precision_pca:.4f}, "
      f"Recall: {recall_pca:.4f}, F1: {f1_pca:.4f}, "
      f"Time: {end_pca_svm - start_pca_svm:.2f}s")

# -----------------------------------------
# 8. Bar plot comparison (Accuracy, Precision, Recall, F1)
# -----------------------------------------
metrics_names = ["Accuracy", "Precision", "Recall", "F1-score"]
full_scores = [accuracy_full, precision_full, recall_full, f1_full]
pca_scores = [accuracy_pca, precision_pca, recall_pca, f1_pca]

x = np.arange(len(metrics_names))
width = 0.35

plt.figure(figsize=(10, 5))
plt.bar(x - width/2, full_scores, width, label='SVM (All Features)')
plt.bar(x + width/2, pca_scores, width, label='SVM (PCA Features)')

plt.xticks(x, metrics_names, rotation=15)
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.title("SVM Performance Before and After PCA")
plt.legend()
plt.tight_layout()
plt.savefig("svm_pca_bar_metrics.png", dpi=300)
plt.show()

# -----------------------------------------
# 9. Total script time end
# -----------------------------------------
overall_end = time.time()
print(f"\nTOTAL script processing time: {overall_end - overall_start:.2f} seconds")
