from sklearn.ensemble import IsolationForest

class Trainer:
    def __init__(self, contamination = 0.01):
        self.model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)

    def train(self, X_train):
        self.model.fit(X_train)
    
    def predict(self, X_test):
        y_pred_raw = self.model.predict(X_test) 
        #sklearn output is 1for normal and -1 for Fraud
        # we need 0 for normal, 1 for fraud
        y_pred = [1 if x==-1 else 0 for x in y_pred_raw]
        return y_pred
    
    # for anomaly score distribution
    def get_scores(self, X_test):
        # decision_function, gives anomaly degree.
        # the more lower the value, higher the anomaly. (i cant english at 2 a.m. :((  )
        return self.model.decision_function(X_test)


