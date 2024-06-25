import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

# Load and preprocess data
data = pd.read_csv("student_data.csv", sep=",")

# Drop unnecessary features
data = data.drop(columns=['reg/no'])

# Convert non-numerical columns to numerical
non_numerical_columns = ['learning_disability', 'attends_tutoring','absences','interest_in_course_studying']

for column in non_numerical_columns:
    data[column] = data[column].astype('category').cat.codes

# Split data into features and target
features = data.drop(columns=['results'])
target = data['results']
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.1, random_state=42)

#model = LinearRegression()
model = KNeighborsRegressor(n_neighbors=5)
#model = RandomForestRegressor(n_estimators=100)

model.fit(x_train, y_train)
prediction = model.predict(x_test)
mse = mean_squared_error(y_test, prediction)
rmse = mean_squared_error(y_test, prediction, squared=False)  # RMSE
mae = mean_absolute_error(y_test, prediction)  # MAE


print("Mean Squared Error (MSE):", mse * 100)
print("Root Mean Squared Error (RMSE):", rmse * 100)
print("Mean Absolute Error (MAE):", mae * 100)
