import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Page title
st.set_page_config(page_title="House Price Prediction", layout="centered")

# Load dataset
df = pd.read_csv("house_prices_practice.csv")

# Select required columns
df = df[["GrLivArea", "BedroomAbvGr", "FullBath", "SalePrice"]]

# Rename columns for easy understanding
df.columns = ["Area", "Bedrooms", "Bathrooms", "Price"]

# Features and target
X = df[["Area", "Bedrooms", "Bathrooms"]]
y = df["Price"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict on training data for R2 score
y_pred = model.predict(X)
score = r2_score(y, y_pred)

# Title
st.title("System")
st.header("House Price Prediction")

# User input
area = st.number_input("Living Area (sqft)", min_value=500, max_value=10000, value=2000)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)

# Prediction button
if st.button("Predict Price"):
    prediction = model.predict([[area, bedrooms, bathrooms]])
    st.success(f"Predicted House Price: {prediction[0]:.2f}")

# Show dataset preview
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Show model accuracy
st.subheader("Model Accuracy")
st.write(f"R2 Score: {score:.2f}")

# Simple chart
st.subheader("Area Graph")
st.line_chart(df["Area"])