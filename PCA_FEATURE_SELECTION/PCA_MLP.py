# pca_mlp.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("creditcard.csv")

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
    stratify=y       # keep the same fraud ratio in train and test
)

# -----------------------------------------
# 3. Scale data (for PCA and MLP)
# -----------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------
# 4. MLP Model – Train on ALL original features
# -----------------------------------------
mlp_full = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)

mlp_full.fit(X_train_scaled, y_train)

y_pred_full = mlp_full.predict(X_test_scaled)
y_prob_full = mlp_full.predict_proba(X_test_scaled)[:, 1]

# Metrics for MLP with all features
auc_full = roc_auc_score(y_test, y_prob_full)
precision_full = precision_score(y_test, y_pred_full)
recall_full = recall_score(y_test, y_pred_full)
f1_full = f1_score(y_test, y_pred_full)

print("\n===== MLP Model 1: Using ALL Features =====")
print("AUC:", auc_full)
print("Precision:", precision_full)
print("Recall:", recall_full)
print("F1:", f1_full)

# -----------------------------------------
# 5. Apply PCA (keep 95% of variance)
# -----------------------------------------
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("\nOriginal number of features:", X_train_scaled.shape[1])
print("Number of PCA features:", X_train_pca.shape[1])

# -----------------------------------------
# 6. MLP Model – Train on PCA features
# -----------------------------------------
mlp_pca = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)

mlp_pca.fit(X_train_pca, y_train)

y_pred_pca = mlp_pca.predict(X_test_pca)
y_prob_pca = mlp_pca.predict_proba(X_test_pca)[:, 1]

# Metrics for MLP with PCA features
auc_pca = roc_auc_score(y_test, y_prob_pca)
precision_pca = precision_score(y_test, y_pred_pca)
recall_pca = recall_score(y_test, y_pred_pca)
f1_pca = f1_score(y_test, y_pred_pca)

print("\n===== MLP Model 2: Using PCA (95% variance) =====")
print("AUC:", auc_pca)
print("Precision:", precision_pca)
print("Recall:", recall_pca)
print("F1:", f1_pca)

# -----------------------------------------
# 7. Comparison
# -----------------------------------------
print("\n===== COMPARISON SUMMARY =====")
print("MLP (All Features) → AUC:", auc_full, " | F1:", f1_full)
print("MLP (PCA Features) → AUC:", auc_pca, " | F1:", f1_pca)

# -----------------------------------------
# 8. Plot the loss curve for both models
# -----------------------------------------
# Plotting training loss curve for both models
plt.figure(figsize=(12, 6))
plt.plot(mlp_full.loss_curve_, label='MLP (All Features)')
plt.plot(mlp_pca.loss_curve_, label='MLP (PCA Features)', linestyle='--')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training Loss Curve for MLP with and without PCA')
plt.legend()
plt.show()
