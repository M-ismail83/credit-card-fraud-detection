import pandas as pd

class Preprocessing:
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.Y_test = None


    def load_data(self):
        self.data = pd.read_csv("creditcard.csv").drop('Time', axis=1)  


    def get_fraud(self):
        return self.data[self.data['Class'] == 1]
    
    def get_normal(self):
        return self.data[self.data['Class'] == 0]
    
    
    def splitting(self):
        normals = self.get_normal()
        frauds = self.get_fraud()
        
        X_normal= normals.drop('Class', axis=1)
        
        X_fraud = frauds.drop('Class', axis=1)
        y_fraud = frauds['Class']

        self.X_train = X_normal.sample(frac=0.8, random_state=42)

        remaining_normals = X_normal.drop(self.X_train.index)

        self.X_test = pd.concat([remaining_normals, X_fraud])

        y_test_normal=pd.Series(0, index=remaining_normals.index)
        self.Y_test=pd.concat([y_test_normal, y_fraud])
