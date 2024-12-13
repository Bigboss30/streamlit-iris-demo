import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Load the dataset
data = pd.read_csv('heart.csv')

# Define features and target variable
X = data.drop('HeartDisease', axis=1)
y = data['HeartDisease']

# Create preprocessor
def create_preprocessor():
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ]), ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']),
            
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'])
        ])
    
    return preprocessor

# Instantiate models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "K-Nearest Neighbors": KNeighborsClassifier()
}

# Preprocess and fit the data
preprocessor = create_preprocessor()
X_train = preprocessor.fit_transform(X)

# Train models
for model_name, model in models.items():
    model.fit(X_train, y)

# Streamlit app
st.title("Heart Disease Prediction")

# Input fields for user data
st.sidebar.header('User Input Parameters')

def user_input_features():
    age = st.sidebar.slider('Age', 29, 77, 54)
    sex = st.sidebar.selectbox('Sex', ('M', 'F'))
    chest_pain_type = st.sidebar.selectbox('Chest Pain Type', ('ATA', 'NAP', 'ASY', 'TA'))
    resting_bp = st.sidebar.slider('RestingBP', 80, 200, 120)
    cholesterol = st.sidebar.slider('Cholesterol', 100, 400, 200)
    fasting_bs = st.sidebar.selectbox('FastingBS', (0, 1))
    max_hr = st.sidebar.slider('MaxHR', 60, 200, 150)
    exercise_angina = st.sidebar.selectbox('ExerciseAngina', ('Y', 'N'))
    oldpeak = st.sidebar.slider('Oldpeak', 0.0, 6.2, 1.0)
    st_slope = st.sidebar.selectbox('ST_Slope', ('Up', 'Flat', 'Down'))
    resting_ecg = st.sidebar.selectbox('RestingECG', ('Normal', 'ST', 'LVH'))
    
    data = {'Age': age,
            'Sex': sex,
            'ChestPainType': chest_pain_type,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': fasting_bs,
            'MaxHR': max_hr,
            'ExerciseAngina': exercise_angina,
            'Oldpeak': oldpeak,
            'ST_Slope': st_slope,
            'RestingECG': resting_ecg}
    features = pd.DataFrame(data, index=[0])
    return features
df = user_input_features()

# Transform user input using the fitted preprocessor
df_processed = preprocessor.transform(df)

# Select model
model_name = st.selectbox("Select Model", list(models.keys()))

# Predict and display results
model = models[model_name]
prediction = model.predict(df_processed)
prediction_proba = model.predict_proba(df_processed)

st.subheader('Prediction')
heart_disease = np.array(['No Heart Disease', 'Heart Disease'])
st.write(heart_disease[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)

# Visualize prediction probabilities
fig, ax = plt.subplots()
ax.barh(heart_disease, prediction_proba[0])
ax.set_xlim([0, 1])
st.pyplot(fig)
