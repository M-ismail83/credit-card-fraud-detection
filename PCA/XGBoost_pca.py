import numpy as np
import pandas as pd
import time  # ✅ NEW: for timing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# ============ TOTAL TIME START ============
overall_start = time.time()

# ============ 1) LOAD DATA ============
df = pd.read_csv("creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

# ============ 2) TRAIN / TEST SPLIT ============
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============ 3) SCALE FEATURES ============
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Helper: search best threshold on validation set (max F1)
def find_best_threshold(probs, y_true):
    best_t = 0.5
    best_f1 = -1.0
    thresholds = np.linspace(0.01, 0.99, 99)
    for t in thresholds:
        y_pred = (probs >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1

# Helper: compute metrics on test set
def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return acc, prec, rec, f1

# =====================================================
# 4) XGBOOST WITHOUT PCA  (BASELINE MODEL)
# =====================================================
X_tr_full, X_val_full, y_tr_full, y_val_full = train_test_split(
    X_train_scaled, y_train, test_size=0.2,
    random_state=42, stratify=y_train
)

neg_full = (y_tr_full == 0).sum()
pos_full = (y_tr_full == 1).sum()
scale_pos_weight_full = neg_full / pos_full
print(f"scale_pos_weight (no PCA) = {scale_pos_weight_full:.2f}")

xgb_full = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.0,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight_full,
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
)

# ✅ Time XGBoost (no PCA)
start_full = time.time()
xgb_full.fit(X_tr_full, y_tr_full)

# --- best threshold on validation ---
val_probs_full = xgb_full.predict_proba(X_val_full)[:, 1]
best_thresh_full, best_f1_full = find_best_threshold(val_probs_full, y_val_full)

# --- evaluate on test set ---
test_probs_full = xgb_full.predict_proba(X_test_scaled)[:, 1]
y_test_pred_full = (test_probs_full >= best_thresh_full).astype(int)
end_full = time.time()

acc_full, prec_full, rec_full, f1_full = compute_metrics(
    y_test, y_test_pred_full
)

print(f"\n[NO PCA] Best threshold on validation = {best_thresh_full:.3f} "
      f"with F1 = {best_f1_full:.4f}")

print("\n==== XGBoost WITHOUT PCA (test, best threshold) ====")
print(f"Accuracy : {acc_full}")
print(f"Precision: {prec_full}")
print(f"Recall   : {rec_full}")
print(f"F1 Score : {f1_full}")
print(f"Processing time (XGB without PCA): {end_full - start_full:.2f} seconds")

# =====================================================
# 5) XGBOOST WITH PCA (95% VARIANCE)
# =====================================================
pca = PCA(n_components=0.95)

# ✅ Time PCA transformation
start_pca = time.time()
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
end_pca = time.time()

print("\nOriginal number of features:", X_train_scaled.shape[1])
print("Number of PCA features     :", X_train_pca.shape[1])
print(f"PCA processing time       : {end_pca - start_pca:.2f} seconds")

X_tr_pca, X_val_pca, y_tr_pca, y_val_pca = train_test_split(
    X_train_pca, y_train, test_size=0.2,
    random_state=42, stratify=y_train
)

neg_pca = (y_tr_pca == 0).sum()
pos_pca = (y_tr_pca == 1).sum()
scale_pos_weight_pca = neg_pca / pos_pca
print(f"scale_pos_weight (PCA) = {scale_pos_weight_pca:.2f}")

xgb_pca = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.0,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight_pca,
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
)

# ✅ Time XGBoost (with PCA)
start_xgb_pca = time.time()
xgb_pca.fit(X_tr_pca, y_tr_pca)

# --- best threshold on validation for PCA model ---
val_probs_pca = xgb_pca.predict_proba(X_val_pca)[:, 1]
best_thresh_pca, best_f1_pca = find_best_threshold(val_probs_pca, y_val_pca)

# --- evaluate on test set ---
test_probs_pca = xgb_pca.predict_proba(X_test_pca)[:, 1]
y_test_pred_pca = (test_probs_pca >= best_thresh_pca).astype(int)
end_xgb_pca = time.time()

acc_pca, prec_pca, rec_pca, f1_pca = compute_metrics(
    y_test, y_test_pred_pca
)

print(f"\n[PCA] Best threshold on validation = {best_thresh_pca:.3f} "
      f"with F1 = {best_f1_pca:.4f}")

print("\n==== XGBoost WITH PCA (test, best threshold) ====")
print(f"Accuracy : {acc_pca}")
print(f"Precision: {prec_pca}")
print(f"Recall   : {rec_pca}")
print(f"F1 Score : {f1_pca}")
print(f"Processing time (XGB with PCA): {end_xgb_pca - start_xgb_pca:.2f} seconds")

# =====================================================
# 6) SUMMARY + BAR PLOT (same style as SVM / MLP)
# =====================================================
print("\n===== COMPARISON SUMMARY (XGBoost) =====")
print("XGB (No PCA) → "
      f"Acc: {acc_full:.4f}, Precision: {prec_full:.4f}, "
      f"Recall: {rec_full:.4f}, F1: {f1_full:.4f}")
print("XGB (PCA 95%) → "
      f"Acc: {acc_pca:.4f}, Precision: {prec_pca:.4f}, "
      f"Recall: {rec_pca:.4f}, F1: {f1_pca:.4f}")

metrics_names = ["Accuracy", "Precision", "Recall", "F1-score"]
full_scores = [acc_full, prec_full, rec_full, f1_full]
pca_scores = [acc_pca, prec_pca, rec_pca, f1_pca]

x = np.arange(len(metrics_names))
width = 0.35

plt.figure(figsize=(10, 5))
plt.bar(x - width/2, full_scores, width, label="XGB (No PCA)")
plt.bar(x + width/2, pca_scores, width, label="XGB (PCA 95%)")

plt.xticks(x, metrics_names, rotation=15)
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.title("XGBoost Performance Before and After PCA")
plt.legend()
plt.tight_layout()
plt.savefig("xgb_pca_bar_metrics.png", dpi=300)
plt.show()

# ============ TOTAL TIME END ============
overall_end = time.time()
print(f"\nTOTAL script processing time: {overall_end - overall_start:.2f} seconds")
