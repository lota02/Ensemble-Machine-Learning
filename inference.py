import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class Inference:
    def __init__(self, model_path, label_encoder_path):
        self.model = joblib.load(model_path)
        self.label_encoder = joblib.load(label_encoder_path)
   
    def preprocess_input(self, input_data):
        # Ensure input_data is a DataFrame
        if not isinstance(input_data, pd.DataFrame):
            input_data = pd.DataFrame(input_data)

        # Convert non-numerical columns to numerical using label encoder
        if self.label_encoder is not None:
            non_numerical_columns = ['learning_disability', 'attends_tutoring']
            for column in non_numerical_columns:
                if column in input_data.columns:
                    input_data[column] = self.label_encoder.transform(input_data[column])

        # Map categorical columns with "Low", "Medium", "High" to numerical values
        categorical_mapping = {
            'absences': {'Low': 0, 'Medium': 1, 'High': 2},
            'interest_in_course_studying': {'Low': 0, 'Medium': 1, 'High': 2}
        }
        for column, mapping in categorical_mapping.items():
            if column in input_data.columns:
                input_data[column] = input_data[column].map(mapping)

        return input_data



    def predict(self, input_data):
    # Preprocess input data
        input_data_preprocessed = self.preprocess_input(input_data)
        # Make predictions
        prediction = self.model.predict(input_data_preprocessed)

        return prediction








 