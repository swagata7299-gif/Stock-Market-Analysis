import pandas as pd
import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import base64

# Define the time range
start = '2010-01-01'
end = '2019-12-31'

# Set background image
def set_background_image(image_path):
    with open(image_path, "rb") as img_file:
        img_data = img_file.read()
    img_b64 = base64.b64encode(img_data).decode()
    st.markdown(
        f"""
        <style>
            .stApp {{
                background-image: url("data:image/png;base64,{img_b64}");
                background-size: cover;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# Call the function to set background image (replace 'background.jpg' with your image path)
set_background_image("C:/Users/ru368/Downloads/back.jpg")

# Streamlit App Title
st.title('Stock Market Analysis')
st.markdown("<h2 style='text-align: center; color: white;'>Predict Stock Prices using Machine Learning</h2>", unsafe_allow_html=True)

# User Input for Stock Ticker
user_input = st.text_input("Enter stock ticker (e.g., 'AAPL')", 'AAPL')

# Fetch Data with Error Handling
try:
    df = yf.download(user_input, start, end)
    
    # Check if the DataFrame is empty
    if df.empty:
        st.error(f"No data found for ticker: {user_input}. Please try a different ticker.")
    else:
        st.subheader('Data from 2010-2019')
        st.write(df.describe())
except Exception as e:
    st.error(f"An error occurred: {e}")

# Styling for Subheaders
st.markdown("<hr style='height:2px;border:none;color:#333;background-color:#333;'/>", unsafe_allow_html=True)

# Add a clean white box for charts
st.markdown("<div style='background-color:white;padding: 20px;border-radius:10px;'>", unsafe_allow_html=True)

st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize =(12,6))
plt.plot(df.Close, label='Closing Price', color='blue', linewidth=2)
plt.plot(df.Open, label='Opening Price', color='orange', linewidth=2)
plt.title("Stock Prices", fontsize=14)
plt.legend(loc='best')
st.pyplot(fig)

st.subheader('Opening Price vs Time Chart')
fig = plt.figure(figsize =(12,6))
plt.plot(df.Open, label='Opening Price', color='green', linewidth=2)
plt.title("Opening Price", fontsize=14)
plt.legend(loc='best')
st.pyplot(fig)

st.subheader('mean100 vs Time Chart')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize =(12,6))
plt.plot(ma100, 'r', label='Mean 100')
plt.plot(df.Close, 'b', label='Closing Price')
plt.title("Mean 100 vs Closing Price", fontsize=14)
plt.legend(loc='best')
st.pyplot(fig)

st.subheader('Comparison of mean100, mean200 and closing price')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize =(12,6))
plt.plot(ma100, 'r', label='Mean 100')
plt.plot(ma200, 'b', label='Mean 200')
plt.plot(df.Close, 'g', label='Closing Price')
plt.title("Mean100, Mean200 vs Closing Price", fontsize=14)
plt.legend(loc='best')
st.pyplot(fig)

# End of the styling box for charts
st.markdown("</div>", unsafe_allow_html=True)

#splitting data
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

print(data_training.shape)
print(data_testing.shape)

scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)

# Load model
model = load_model('keras_model.h5')

# Testing part
past_100_days = data_training.tail(100)

# Combine the DataFrames
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler.scale_

scale_factor = 1 / scaler.scale_[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Final graph
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label='Original Price', linewidth=2)
plt.plot(y_predicted, 'r', label='Predicted Price', linewidth=2)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.title("Predictions vs Original Prices", fontsize=14)
plt.legend(loc='best')
st.pyplot(fig2)
