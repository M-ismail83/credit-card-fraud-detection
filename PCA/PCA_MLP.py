# pca_mlp.py
import pandas as pd
import numpy as np  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("PCA/creditcard.csv")

# Split into features (X) and labels (y)
X = df.drop("Class", axis=1)
y = df["Class"]

# -----------------------------------------
# 2. Train-test split
# -----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# -----------------------------------------
# 3. Scale data
# -----------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# pca_mlp.py
import pandas as pd
import numpy as np  
import time  # ✅ NEW: for timing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib.pyplot as plt

# ============ TOTAL TIME START ============
overall_start = time.time()

# Load the dataset
df = pd.read_csv("PCA/creditcard.csv")

# Split into features (X) and labels (y)
X = df.drop("Class", axis=1)
y = df["Class"]

# -----------------------------------------
# 2. Train-test split
# -----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# -----------------------------------------
# 3. Scale data
# -----------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------
# 4. MLP Model – Train on ALL original features
# -----------------------------------------
mlp_full = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)

# ⏱️ Time MLP without PCA
start_full = time.time()
mlp_full.fit(X_train_scaled, y_train)
y_pred_full = mlp_full.predict(X_test_scaled)
end_full = time.time()

# Metrics for MLP with all features
acc_full = accuracy_score(y_test, y_pred_full)
precision_full = precision_score(y_test, y_pred_full, zero_division=0)
recall_full = recall_score(y_test, y_pred_full, zero_division=0)
f1_full = f1_score(y_test, y_pred_full, zero_division=0)

print("\n===== MLP Model 1: Using ALL Features =====")
print("Accuracy :", acc_full)
print("Precision:", precision_full)
print("Recall   :", recall_full)
print("F1       :", f1_full)
print(f"Processing time (MLP without PCA): {end_full - start_full:.2f} seconds")

# -----------------------------------------
# 5. Apply PCA (keep 95% of variance)
# -----------------------------------------
pca = PCA(n_components=0.95)

# ⏱️ Time PCA step
start_pca = time.time()
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
end_pca = time.time()

print("\nOriginal number of features:", X_train_scaled.shape[1])
print("Number of PCA features:", X_train_pca.shape[1])
print(f"PCA processing time: {end_pca - start_pca:.2f} seconds")

# -----------------------------------------
# 6. MLP Model – Train on PCA features
# -----------------------------------------
mlp_pca = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)

# ⏱️ Time MLP with PCA
start_mlp_pca = time.time()
mlp_pca.fit(X_train_pca, y_train)
y_pred_pca = mlp_pca.predict(X_test_pca)
end_mlp_pca = time.time()

# Metrics for MLP with PCA features
acc_pca = accuracy_score(y_test, y_pred_pca)
precision_pca = precision_score(y_test, y_pred_pca, zero_division=0)
recall_pca = recall_score(y_test, y_pred_pca, zero_division=0)
f1_pca = f1_score(y_test, y_pred_pca, zero_division=0)

print("\n===== MLP Model 2: Using PCA (95% variance) =====")
print("Accuracy :", acc_pca)
print("Precision:", precision_pca)
print("Recall   :", recall_pca)
print("F1       :", f1_pca)
print(f"Processing time (MLP with PCA): {end_mlp_pca - start_mlp_pca:.2f} seconds")

# -----------------------------------------
# 7. Comparison (text)
# -----------------------------------------
print("\n===== COMPARISON SUMMARY =====")
print("MLP (All Features) → "
      f"Acc: {acc_full:.4f}, Precision: {precision_full:.4f}, "
      f"Recall: {recall_full:.4f}, F1: {f1_full:.4f}")
print("MLP (PCA Features) → "
      f"Acc: {acc_pca:.4f}, Precision: {precision_pca:.4f}, "
      f"Recall: {recall_pca:.4f}, F1: {f1_pca:.4f}")

# -----------------------------------------
# 8. Plot the loss curve for both models
# -----------------------------------------
plt.figure(figsize=(12, 5))
plt.plot(mlp_full.loss_curve_, label='MLP (All Features)')
plt.plot(mlp_pca.loss_curve_, label='MLP (PCA Features)', linestyle='--')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training Loss Curve for MLP with and without PCA')
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------------------
# 9. Plot metric comparison (Accuracy, Precision, Recall, F1)
# -----------------------------------------
metrics_names = ["Accuracy", "Precision", "Recall", "F1-score"]
full_scores = [acc_full, precision_full, recall_full, f1_full]
pca_scores = [acc_pca, precision_pca, recall_pca, f1_pca]

x = np.arange(len(metrics_names))
width = 0.35  

plt.figure(figsize=(10, 5))
plt.bar(x - width/2, full_scores, width, label='MLP (All Features)')
plt.bar(x + width/2, pca_scores, width, label='MLP (PCA Features)')

plt.xticks(x, metrics_names)
plt.ylabel('Score')
plt.ylim(0, 1.05)
plt.title('MLP Performance Before and After PCA')
plt.legend()
plt.tight_layout()
plt.savefig("PCA/mlp_pca_metrics.png")
plt.show()

# ============ TOTAL TIME END ============
overall_end = time.time()
print(f"\nTOTAL script processing time: {overall_end - overall_start:.2f} seconds")
