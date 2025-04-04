import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('apartment_price_data.csv')
df.head()

#spliting feature from target
x = df.drop('price', axis=1)
y = df['price']

# spliting the data into test and train
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=42)

lr = LinearRegression()
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

mae_lr = mean_absolute_error(y_test, y_pred)
mse_lr = mean_squared_error(y_test, y_pred)
r2_lr = r2_score(y_test, y_pred)

print('Linear Regression Performance')
print(f'MAE: {mae_lr: .2f}')
print(f'MSE: {mse_lr: .2f}')
print(f'R^2: {r2_lr: .2f}')

### Streamlit App
import streamlit as st

#Function to load the model
def load_model():
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model

# Function to work prediction based on user input
def  predict_price(model, num_rooms, parking_space, width, num_convenience, building_age, distance_to_center, location):
    # Prepare the input data as a DataFrame with the same column names
    input_data = pd.DataFrame({
        'num_rooms' : [num_rooms],
        'parking_space' : [parking_space],
        'width' : [width],
        'num_conveniences' : [num_convenience],
        'building_age' : [building_age],
        'distance_to_center' : [distance_to_center],
        'location' : [location]
    })
    prediction = model.predict(input_data)
    return prediction[0]

# Calling the model function (load model)
model = load_model()

## Web App Interface
st.title('Apartment Price Prediction')
st.subheader('This app predicts the price of an apartment or building based on certain features')

# Input features for users to enter
num_rooms = st.slider('Number of Rooms', 1, 6, 3)
parking_space = st.selectbox('Parking Space', [0, 1, 2])
width = st.slider('Width in square meters', 30, 300, 100)
num_conveniences = st.slider('Number of conveniences', 1, 4, 2 )
building_age = st.slider('Building Age in years', 1, 50, 10)
distance_to_center = st.slider('DIstance in km', 1, 50, 10)
location =st.selectbox('Location (1 = City center, 5= outskirts)', [1, 2, 3, 4, 5])

# Button to trigger the prediction
if st.button("Predict Price"):
    predicted_price = predict_price(model, num_rooms, parking_space, width, num_conveniences, building_age, distance_to_center, location)
    st.write(f"Predicted Price: ${predicted_price:,.2f}")