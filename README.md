# Credit Card Fraud Detection
The repository of a Machine Learning project for the course SE3007.




## About This Project

This project aims to distinguish whether a credit card transaction is fraud or authentic. Our machine learning methods had to perform through unique challanges because of the highly imbalanced dataset. This makes the task challenging because models must detect rare positive cases without producing too many false positives. As a team, we implemented 4 different machine learning methods:

- SMOTE (Synthetic Minority Over Sampling) Technique
- PCA (Principal Component Analysis) Technique
- Missing Value Imputation Technique
- Outlier Technique




## About The Dataset

We used the [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), from Kaggle. The dataset has logs of credit card transactions which can either be normal or fraud.

- Total Transactions : 284,807
- Fraud Transactions : 492 (0.172% of total)
- Format : numerical
- Total Features : 30 (background information only available for Time and Amount, because of confidentiality)


-----



## SMOTE (Synthetic Minority Over Sampling) Technique


### Methodology

### Results
Our models show auspicious results, even in the inbalanced dataset conditions.

-----


## PCA (Principal Component Analysis) Technique

This part of the project investigates the impact of dimensionality reduction using PCA on different machine learning models for credit card fraud detection.

Instead of assuming that PCA always improves performance, our objective was to systematically evaluate when PCA helps, when it hurts, and why, using the same experimental pipeline for all models.

### Methodology
To ensure a fair comparison, the following workflow was applied both with and without PCA for each model:

1. Data Splitting

 - The dataset was split into 80% training and 20% testing sets.
 - Stratified sampling was used to preserve the original class imbalance.

2. Feature Scaling

- All features were standardized using StandardScaler.
- The scaler was fitted on the training set and applied to the test set.

3. PCA Transformation

 - PCA was applied with 95% variance retention.
 - This reduced the feature space from 30 original features to 27 principal components.
 - PCA was fitted on the training data and applied to the test data to avoid data leakage.

4. Model Training
For each algorithm (XGBoost, MLP, SVM, KNN), two models were trained:

 - One using the original scaled features
 - One using the PCA-transformed features

5. Threshold Optimization (XGBoost only)
 
 - For XGBoost, an internal training/validation split was used.
 - The decision threshold was tuned to maximize the F1-score, improving performance on the imbalanced dataset.

6. Evaluation Metrics
All models were evaluated on the test set using:
 - Accuracy
 - Precision
 - Recall
 - F1-score



### Model-by-Model Analysis

#### XGBoost 
 - PCA caused a small decrease in precision, recall, and F1-score.
 - Computational efficiency slightly improved.

This behavior is expected because XGBoost relies on original feature interactions and decision-tree splits. PCA mixes features into linear components, which:
 - Reduces interpretability for trees
 - Weakens informative fraud-related patterns
 - Removes small but important signals

As a result, PCA slightly harms XGBoost performance.


#### Multi-Layer Perceptron (MLP)
 - PCA increased recall but reduced precision, leading to a small change in F1-score.

PCA benefits MLP by:
 - Reducing noise
 - Simplifying the feature space

However, the loss of fine-grained feature details increases false positives, creating a recall–precision trade-off.

#### Support Vector Machine (SVM)
 - Performance degraded significantly after PCA, especially in recall and F1-score.

SVM is highly sensitive to:
 - Feature-space geometry
 - Distance relationships
 - Class overlap

PCA alters these relationships, and when combined with extreme class imbalance, results in weaker margins and reduced class separability.

#### K-Nearest Neighbors (KNN)
After PCA, KNN showed:
 - Slightly higher precision
 - Slightly lower recall
 - Nearly unchanged F1-score

 PCA reduced noise and improved distance stability, but also slightly altered neighborhood relationships.
The reduced dimensionality improved computational efficiency without significantly affecting overall performance.

Key Takeaways from PCA Experiments
 - PCA does not universally improve model performance
 - Tree-based models (XGBoost) are negatively affected by PCA
 - Distance-based and neural models show mixed effects
 - PCA mainly improves efficiency and noise handling, not necessarily accuracy
 - The usefulness of PCA strongly depends on the model architecture and decision mechanism

-----


## Missing Value Imputation Technique

### Methodology

### Results
Our models show auspicious results, even in the inbalanced dataset conditions.

-----


## Outlier Technique

### Methodology

In the Outlier Method, we trained the model only on the normal transactions. Our objective was to make the model learn the normals very well, and it sees frauds as foreign/abnormal on testing.
Isolation Forest was the perfect fit for this purpose.

### Results
Our models show auspicious results, even in the imbalanced dataset conditions. 

- Anomaly Score Distribution
![Anomaly-Score-Distribution](anomaly_score_distribution.png)

Less overlap means model is doing better at classifying.


- Confusion Matrix
![Confusion-Matrix](confusion_matrix.png)

What we see here is the amounts of; normal transactions flagged as normal, fraud transactions flagged as normal, fraud transactions flagged as fraud, normal transactions flagged as frauds.


- tSNE Visualization
![tSNE-Visualization](tSNE_visualization.png)

Just a fun visualization, may be deactivated in the evaluation.py


- Precision-Recall Curve
![Precision-Recall](precision_recall.png)

In ideal conditions, this graph would start small and rise towards the end (basically the opposite of this very graph). This is because our model focuses on the very obvious frauds in the beginning, and the model gets "more paranoid" as the frauds get more subtle. We can improve these results by doing feature engineering on our parameters, which are: 

#### Future Work & Potential Improvements

In this project we focused on a baseline using Isolation Forest. However, the following steps could further improve the model's recall and precision:

1.  Temporal Analysis: Converting the raw 'Time' seconds into "Hour of the Day" to capture time-dependent fraud patterns (e.g., late-night transactions).
2.  Log-Scaling Amount: Applying Logarithmic Transformation to the 'Amount' feature to handle its heavy-tailed distribution, making it easier for the model to process extreme values.
3.  Feature Interactions: Generating new features by multiplying 'Amount' with key PCA components (e.g., 'Amount * V14') to highlight correlations specific to fraud.
4.  Ensemble Methods: Combining Isolation Forest with Local Outlier Factor (LOF) or Deep Autoencoders in a voting classifier to create a more robust hybrid detection system.
