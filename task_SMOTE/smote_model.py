from imblearn.over_sampling import SMOTE
from collections import Counter

class SMOTEModel:
    def __init__(self, sampling_strategy='auto', random_state = 42):
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.smote = SMOTE(sampling_strategy=self.sampling_strategy, random_state=self.random_state)
    
    def fit_resample(self, X, y):
        resamples = self.smote.fit_resample(X, y)
        
        X_resampled = resamples[0]
        y_resampled = resamples[1]
        
        return X_resampled, y_resampled
    
    def set_sampling_strategy(self, target_count, normal_count = 5000):
        ratio = target_count / normal_count
        
        self.sampling_strategy = ratio
        
        self.smote = SMOTE(sampling_strategy=self.sampling_strategy, random_state=self.random_state)
        print(f"New normal/fraud ratio: {ratio:.3f}, new fraud count: {target_count}")
    
    def get_class_distribution(self, y):
        count = Counter(y)
        return count