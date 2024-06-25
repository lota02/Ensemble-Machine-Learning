import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
import joblib 

class Model:
    def __init__(self, data: pd, test_size: float, features, target): 
        self.data = data
        self.test_size = test_size
        self.features = features
        self.target = target
        self.scaler = StandardScaler()
        self.label_encoder = None 
      
    def preprocess_data(self):
        # Apply scaling to features
        self.features = self.scaler.fit_transform(self.features)

    def create_base_models(self):
        # Create individual regression models
        model1 = LinearRegression()
        model2 = KNeighborsRegressor(n_neighbors=5)
        model3 = RandomForestRegressor(n_estimators=100)
        return [('linear_regression', model1), ('knn_regressor', model2), ('random_forest_regressor', model3)]

    def create_ensemble(self, base_models):
        # Create VotingRegressor ensemble
        ensemble = VotingRegressor(estimators=base_models)
        return ensemble

    def evaluate_model(self, model, x_test, y_test):
        # Calculate mean squared error
        prediction = model.predict(x_test)
        mse = mean_squared_error(y_test, prediction)
        return mse

    def train(self):
        mse = float('inf')  # Assign a default value to mse
        for _ in range(15):
            try:
                # Split the data
                x_train, x_test, y_train, y_test = train_test_split(
                    self.features, self.target, test_size=self.test_size, random_state=42
                )
                
             
                self.preprocess_data()
                # Create base models
                base_models = self.create_base_models()

                # Train base models
                for name, model in base_models:
                    model.fit(x_train, y_train)

                # Create ensemble
                base_models = self.create_base_models()
                self.ensemble = self.create_ensemble(base_models)

                # Train ensemble
                self.ensemble.fit(x_train, y_train)

                # Evaluate ensemble using Mean Squared Error (MSE)
                mse = self.evaluate_model(self.ensemble, x_test, y_test)

                if mse <= 0.15: 
                    path = '/Users/Lota/Documents/Ensemble'
                    joblib.dump(self.label_encoder, os.path.join(path, 'label_encoder.joblib'))
                    joblib.dump(self.ensemble, os.path.join(path, 'trained_model.joblib'))
                    return mse
            except Exception as e:
                print("Error occurred during training:", e)
        return mse
