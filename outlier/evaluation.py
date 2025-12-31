import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

class Evaluator:
    def __init__(self):
        pass

    def print_report(self, y_true, y_pred):
        print("\n--- Classification Report ---")
        # target_names=['Normal', 'Fraud']
        print(classification_report(y_true, y_pred, target_names=['Normal', 'Fraud']))

    # confusion matrix
    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='cool', cbar=False)
        plt.title('Confusion Matrix (Normal vs Fraud)')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

    # anomaly score distribution
    def plot_score_distribution(self, y_true, y_scores):
        plt.figure(figsize=(8, 6))
        
        # seperate nromal and fraud
        normal_scores = y_scores[y_true == 0]
        fraud_scores = y_scores[y_true == 1]
        
        # Histograms
        plt.hist(normal_scores, bins=50, alpha=0.6, color='blue', label='Normal', density=True)
        plt.hist(fraud_scores, bins=50, alpha=0.6, color='red', label='Fraud', density=True)
        
        plt.title('Anormaly score dist.')
        plt.xlabel('Anormaly score (Lower = Less abnormal)')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.show()

        # precision recall
    def plot_pr_curve(self, y_true, y_scores):
        from sklearn.metrics import precision_recall_curve, auc
        
        # Inverts scores (Isolation Forest low score = fraud)
        # PR curve expects high score as positive, thats why we invert.
        y_scores_inverted = -y_scores
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores_inverted)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(6, 6))
        plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR Curve (AUC = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="best")
        plt.grid(True)
        plt.show()

        # tSNE
    def plot_tsne(self, X_test, y_true, y_pred):
        from sklearn.manifold import TSNE
        import numpy as np
                
        # onnly some part of the data (for speed)
        n_samples = 5000
        if len(X_test) > n_samples:
             # includes both normals and frauds
            X_sample = X_test.sample(n=n_samples, random_state=42)
            y_true_sample = y_true[X_sample.index]
            y_pred_sample = pd.Series(y_pred, index=X_test.index)[X_sample.index]
        else:
            X_sample = X_test
            y_true_sample = y_true
            y_pred_sample = y_pred

        # t-SNE 
        tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
        X_2d = tsne.fit_transform(X_sample)
        
        # visualization
        plt.figure(figsize=(10, 8))
        colors = ['blue', 'red']
        labels = ['Normal', 'Fraud']
        
        for i in range(2):
            plt.scatter(X_2d[y_true_sample == i, 0], X_2d[y_true_sample == i, 1], 
                        c=colors[i], label=f'True {labels[i]}', alpha=0.5, s=20)
            
        plt.title('t-SNE visualization in 2D')
        plt.legend()
        plt.grid(True)
        plt.show()