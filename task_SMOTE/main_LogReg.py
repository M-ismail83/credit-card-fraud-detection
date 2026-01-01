import pandas as pd
import joblib
import matplotlib.pyplot as plt
from dataset_handler import dataset_handler
from smote_model import SMOTEModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

# Dataset reading and splitting into train and test
whole_set = pd.read_csv("task_SMOTE/creditcard.csv")
handler = dataset_handler(dataframe = whole_set)

X, y = handler.get_features_and_labels(handler.data)

X_train_pool, X_test, y_train_pool, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

ds_pool = pd.concat([X_train_pool, y_train_pool], axis=1)

# Get all the frauds and normals
pool_normals = ds_pool[ds_pool['Class'] == 0]
pool_frauds = ds_pool[ds_pool['Class'] == 1]

# Select a number of fraud and normals for training
selected_normals = pool_normals.sample(n=5000, random_state=42)
selected_frauds = pool_frauds.sample(n=350, random_state=42)

df_train_final = pd.concat([selected_normals, selected_frauds])
df_train_final = df_train_final.sample(frac=1, random_state=42).reset_index(drop=True)

X_train = df_train_final.drop(['Class', 'Time'], axis=1)
y_train = df_train_final['Class']

X_loop_train = X_train
y_loop_train = y_train

fraud_count = y_train[y_train == 1].count()

# Training and testing 

smote_model = SMOTEModel(random_state=42)

acc = {}
recall = {}
f1 = {}
prec = {}

while fraud_count <= 4000:
    
    print(f"{X_loop_train.shape}")
    print(f"Model with fraud count: {fraud_count}")
    model_LogReg = LogisticRegression(solver='lbfgs', max_iter=10000)
    model_LogReg.fit(X_loop_train, y_loop_train)
    y_pred = model_LogReg.predict(X_test)
    
    # Accuracy
    print(accuracy_score(y_test, y_pred))
    acc[f"Fraud: {fraud_count}"] = accuracy_score(y_test, y_pred)
    
    # Recall
    print(recall_score(y_test, y_pred))
    recall[f"Fraud: {fraud_count}"] = recall_score(y_test, y_pred)
    
    # F1 Score
    print(accuracy_score(y_test, y_pred))
    f1[f"Fraud: {fraud_count}"] = f1_score(y_test, y_pred)
    
    # Precision
    print(accuracy_score(y_test, y_pred))
    prec[f"Fraud: {fraud_count}"] = precision_score(y_test, y_pred)
    
    filename = f"task_SMOTE/models/LogReg/model_fraud_{fraud_count}.joblib"
    joblib.dump(model_LogReg, filename)
    
    new_fraud_count = fraud_count + 350
    
    if fraud_count > 4000:
        fraud_count = 4000
    
    smote_model.set_sampling_strategy(new_fraud_count)
    X_loop_train, y_loop_train = smote_model.fit_resample(X_train, y_train)
    
    fraud_count += 350
    if fraud_count > 4000:
        break
    
def prepare_plot_data(metrics_dict):
    x_values = [int(k.split(': ')[1]) for k in metrics_dict.keys()]
    y_values = list(metrics_dict.values())
    
    combined = sorted(zip(x_values, y_values))
    x_sorted = [c[0] for c in combined]
    y_sorted = [c[1] for c in combined]
    
    return x_sorted, y_sorted
    
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

metrics_data = [
    (acc, 'Accurac', axes[0, 0], 'blue'),
    (recall, 'Recall', axes[0, 1], 'green'),
    (prec, 'Precision', axes[1, 0], 'red'),
    (f1, 'F1 Score', axes[1, 1], 'purple')
]

for data_dict, title, ax, color in metrics_data:
    x, y = prepare_plot_data(data_dict)
    
    ax.plot(x, y, marker='o', linestyle='-', color=color, linewidth=2, markersize=6)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Fraud Count')
    ax.set_ylabel('Score')
    
plt.savefig("task_SMOTE/plots_and_figures/LogReg.png")
plt.show()
    
    
    
    