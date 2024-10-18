import streamlit as st
from model import Model
from inference import Inference
from database import connect_to_database, create_table, insert_prediction
import pandas as pd
import random
from pandas import DataFrame
import numpy as np
from sklearn import preprocessing
import joblib

# Load and preprocess data
data = pd.read_csv("student_data.csv", sep=",")

# Drop unnecessary features
data = data.drop(columns=['reg/no'])

# Convert non-numerical columns to numerical
le = preprocessing.LabelEncoder()                
non_numerical_columns = ['learning_disability', 'attends_tutoring','absences','interest_in_course_studying']

for column in non_numerical_columns:
    data[column] = le.fit_transform(data[column])

# Split data into features and target
features = data.drop(columns=['results'])
target = data['results']

#Define paths to model files
model_path = '/Users/Lota/Documents/Ensemble/trained_model.joblib'
label_encoder_path = '/Users/Lota/Documents/Ensemble/label_encoder.joblib'

# Create an instance of Inference with the specified paths
inference_model = Inference(model_path, label_encoder_path)

# Connect to the database
conn = connect_to_database()

# Create the predictions table if it doesn't exist
create_table(conn)


def determine_honor(prediction):
    # Determine which honours the predictions falls under 
    if prediction >= 4.50:
        return "First Class Honor"
    elif 3.50 <= prediction < 4.50:
        return "Second Class Upper"
    elif 2.40 <= prediction < 3.50:
        return "Second Class Lower"
    elif 1.50 <= prediction < 2.40:
        return "Third Class"
    elif 1.00 <= prediction < 1.50:
        return "Pass"
    else:
        return "Fail"


# Define the Streamlit app
def main():
    st.title('Student Performance Prediction')

    # Sidebar for input details
    st.sidebar.markdown("<h2 style='font-size:30px;'>Input Details</h2>", unsafe_allow_html=True)

    student_name = st.sidebar.text_input('Name', value="", help="Enter your name")  

    st.sidebar.write("---")

    age = st.sidebar.number_input('Age', value=None, min_value=None, max_value=100, step=None, help="Enter your age")

    st.sidebar.write("---")

    post_utme = st.sidebar.number_input('Post UTME Score', value=None, min_value=0, max_value=400, step=None, help="Enter your Post UTME score (0-400)")

    st.sidebar.write("---")

    waec_average = st.sidebar.number_input('WAEC Average', value=None, min_value=0, max_value=100, step=None, help="Enter your WAEC average (0-100)")

    st.sidebar.write("---")

    study_hours = st.sidebar.number_input('Weekly Study Hours', value=None, min_value=0, max_value=168, step=None, help="Enter the number of study hours per week")

    st.sidebar.write("---")

    gpa = st.sidebar.number_input('GPA', value=None, min_value=0.0, max_value=5.0, step=None, help="Enter your current GPA")

    st.sidebar.write("---")

    learning_disability = st.sidebar.selectbox('Learning Disability', ['No', 'Yes'], help="Do you have a learning disability?")

    st.sidebar.write("---")

    attends_tutoring = st.sidebar.selectbox('Attends Tutoring', ['No', 'Yes'], help="Do you attend tutoring?")

    st.sidebar.write("---")

    absences = st.sidebar.selectbox('Abscences', ['Low', 'Medium', 'High'], help="How involved are you in school activities? ")

    st.sidebar.write("---")

    interest_in_course_studying = st.sidebar.selectbox('Interest in Course Studying', ['Low', 'Medium', 'High'], help="Select your interest level in studying the course")

    # Button for prediction
    if st.sidebar.button('Predict', key='predict_button', help="Click here to make a prediction"):
        # Check if all inputs are provided
        if None in [student_name, age, post_utme, waec_average, study_hours, gpa, learning_disability, attends_tutoring,
                    absences, interest_in_course_studying]:
            st.warning('Please insert all inputs before making a prediction.')
        else:
            # Preprocess input data
            input_data = pd.DataFrame({
                "age": [age], "post_utme": [post_utme], "waec_average": [waec_average],
                "study_hours": [study_hours], "gpa": [gpa],
                "learning_disability": [1 if learning_disability == 'Yes' else 0],
                "attends_tutoring": [1 if attends_tutoring == 'Yes' else 0],
                "absences": [absences], "interest_in_course_studying": [interest_in_course_studying]
            })
            # Make prediction
            prediction = inference_model.predict(input_data)[0]  # Extracting the float value

            # Display prediction result
            st.subheader('Prediction Result')
            st.write(f"**Student Name:** {student_name}")
            st.write(f"**Predicted GPA:** {prediction:.2f}")

            # Determine honor based on the predicted GPA
            honor = determine_honor(prediction)
            st.write(f"**Predicted Honor:** <span style='color: green;'>{honor}</span>", unsafe_allow_html=True)

            # Store prediction in the database
            insert_prediction(conn, student_name, prediction, gpa)

           

# Run the Streamlit app
if __name__ == "__main__":
    main()












