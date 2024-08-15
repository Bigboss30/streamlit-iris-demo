import streamlit as st
import pandas as pd
import joblib
from sklearn.datasets import load_iris
# Load the model
model = joblib.load('iris_model.pkl')

# Load the iris dataset
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['Species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Streamlit app
st.title('Iris Dataset')

# Display the first few rows of the dataset
st.dataframe(iris_df.head())

# Define a function for prediction
def predict_species(features):
    prediction = model.predict([features])
    species = iris.target_names[prediction][0]
    return species

# Streamlit app
st.title('Iris Flower Species Prediction')

# Input fields for the features
sepal_length = st.number_input('Sepal Length (cm)', min_value=0.0, max_value=10.0, step=0.1)
sepal_width = st.number_input('Sepal Width (cm)', min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.number_input('Petal Length (cm)', min_value=0.0, max_value=10.0, step=0.1)
petal_width = st.number_input('Petal Width (cm)', min_value=0.0, max_value=10.0, step=0.1)

# Predict button
if st.button('Predict'):
    features = [sepal_length, sepal_width, petal_length, petal_width]
    prediction = predict_species(features)
    st.write(f'The predicted species is: **{prediction}**')


st.sidebar.header('Hello *World!* :sunglasses:')
