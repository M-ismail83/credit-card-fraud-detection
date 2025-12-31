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

To study how dimensionality reduction affects model behavior, we applied Principal Component Analysis (PCA) using 95% variance retention, which reduced the dataset from 30 original features to 27 principal components. We then compared the performance of several machine learning algorithms before and after PCA using the exact same workflow.
The goal of this report is to analyze how PCA influences accuracy, precision, recall, and F1-score, and to explain why each model’s performance improved, decreased, or stayed similar after dimensionality reduction.

### Methodology

We used the same workflow for all algorithms to ensure a fair comparison.
1. Data Loading & Splitting:
The dataset was loaded from creditcard.csv. Features (X) and labels (y) were separated, then split into 80% training and 20% testing using stratified sampling.


2. Feature Scaling:
All features were standardized using StandardScaler (fit on training data, applied to test data).


3. PCA Transformation (95% Variance):
PCA was applied to the scaled training data, reducing 30 features → 27 components. The same transformation was applied to the test set.


4. Model Training (Before and After PCA): 
For each algorithm (XGBoost, MLP, SVM, KNN), we trained:
- A model on the original scaled features
- A model on the PCA-reduced features


5. Validation Threshold Selection: 
The training data was internally split again into training/validation subsets.
The validation set was used to find the best decision threshold that maximized the F1-score.

6. Model Evaluation:
Each model was evaluated on the test set using:
Accuracy, Precision, Recall, F1-score, Processing time

This allowed us to directly compare how PCA affects performance and computation across different algorithms.


### Model-by-Model Results & Analysis

#### XGBoost Results
After applying PCA, XGBoost showed a small decrease in performance across precision, recall, and F1-score. Processing time improved slightly due to fewer input features.

XGBoost relies heavily on original feature interactions and raw variable splits. PCA mixes these features into new components, which:
- Reduces interpretability for the trees
- Weakens useful fraud-related patterns
- Removes small-but-important signals
Therefore, PCA slightly harms XGBoost performance.

#### Multi-Layer Perceptron (MLP) Results
#### Support Vector Machine (SVM) Results
#### K-Nearest Neighbors (KNN) Results

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