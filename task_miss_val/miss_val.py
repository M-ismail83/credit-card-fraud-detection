import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import average_precision_score, roc_auc_score

dataset = pd.read_csv("task_miss_val/creditcard.csv")

frauds = dataset[dataset['Class'] == 1]
normals = dataset[dataset['Class'] == 0]

normal_sample = normals.sample(n=28000, random_state=42)

X_reduced = pd.concat([frauds, normal_sample])
X_reduced = X_reduced.sample(frac=1, random_state=42).reset_index(drop=True)

X = X_reduced.drop(['Class', 'Time'], axis=1)
y = X_reduced['Class']

X_missing = X.copy()
np.random.seed(42)
mask = np.random.rand(*X_missing.shape) < 0.15
X_missing[mask] = np.nan

X_train, X_test, y_train, y_test = train_test_split(X_missing, y, test_size=0.3, random_state=42, stratify=y)

X_train_clean, X_test_clean, _, _ = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

pipe_median = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler(),
    LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42)
)

pipe_mice = make_pipeline(
    IterativeImputer(max_iter=10, random_state=42),
    StandardScaler(),
    LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42)
)

pipe_knn = make_pipeline(
    KNNImputer(n_neighbors=3),
    StandardScaler(),
    LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42)
)

pipe_tail = make_pipeline(
    SimpleImputer(strategy='constant', fill_value=-999),
    StandardScaler(),
    LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42)
)

pipe_benchmark = make_pipeline(
    StandardScaler(),
    LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42)
)

experiments = {
    "1. Median Imputation": (pipe_median, X_train, X_test),
    "2. MICE Imputation":   (pipe_mice,   X_train, X_test),
    "3. k-NN Imputation":   (pipe_knn,    X_train, X_test),
    "4. End-of-Tail":       (pipe_tail, X_train, X_test),
    "5. Original Data":     (pipe_benchmark, X_train_clean, X_test_clean)
}

print(f"{'Method':<22} | {'AUPRC (Precision-Recall)':<25} | {'ROC-AUC':<10}")
print("-" * 65)

def experimenter():
    results = []
    for name, (model, train_data, test_data) in experiments.items():
        print(f"Running {name}...")
        
        start_time = time.time()
        
        # Train
        model.fit(train_data, y_train)
        training_time = time.time() - start_time
        
        # Predict Probabilities
        y_prob = model.predict_proba(test_data)[:, 1]
        
        # Evaluate
        auprc = average_precision_score(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        
        results.append(
            {
                'Method': name,
                'AUPRC': auprc,
                'ROC-AUC': auc
            }
        )
        
        
        print(f"Done. {name:<22} | {auprc:.4f} {'Original Data Test' if name == 'Original Data' else ''} | {auc:.4f}")
        print(f"Time passed: {training_time:.2f}")
        print("-" * 65)
        
    return results
