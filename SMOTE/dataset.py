import pandas as pd

class dataset:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        return self.data
    
    def get_fraud_data(self):
        return self.data[self.data['Class'] == 1]
    
    def get_normal_data(self):
        return self.data[self.data['Class'] == 0]
    
    def get_normal_sample(self, n_samples):
        normal_data = self.get_normal_data()
        return normal_data.sample(n=n_samples, random_state=42)
    
    def get_fraud_sample(self, n_samples):
        fraud_data = self.get_fraud_data()
        return fraud_data.sample(n=n_samples, random_state=42)
    
    def concat_data(self, data1, data2):
        return pd.concat([data1, data2])
    
    def get_features_and_labels(self, data):
        X = data.drop('Class', axis=1)
        y = data['Class']
        return X, y