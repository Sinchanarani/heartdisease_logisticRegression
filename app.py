import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load and preprocess data
df = pd.read_csv("heart.csv")
history_mapping = {'Absent': 0, 'Present': 1}
df["famhist"] = df["famhist"].map(history_mapping)

# Model training
X = df[['tobacco', 'ldl', 'adiposity', 'famhist', 'typea', 'obesity', 'alcohol', 'age']].values
y = df[['chd']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
model = LogisticRegression(C=100, penalty='l2', solver='liblinear')
model.fit(X_train, y_train)

# Sidebar for user inputs
st.sidebar.header('Input Features')

def user_input_features():
    tobacco = st.sidebar.slider('Tobacco Usage', 0.0, 10.0, 1.0)
    ldl = st.sidebar.slider('LDL Cholesterol', 0.0, 15.0, 3.0)
    adiposity = st.sidebar.slider('Adiposity', 0.0, 50.0, 20.0)
    famhist = st.sidebar.selectbox('Family History', ('Absent', 'Present'))
    typea = st.sidebar.slider('Type A Personality', 0, 100, 50)
    obesity = st.sidebar.slider('Obesity', 0.0, 50.0, 25.0)
    alcohol = st.sidebar.slider('Alcohol Consumption', 0.0, 100.0, 10.0)
    age = st.sidebar.slider('Age', 20, 80, 40)

    famhist = 1 if famhist == 'Present' else 0

    data = {
        'tobacco': tobacco,
        'ldl': ldl,
        'adiposity': adiposity,
        'famhist': famhist,
        'typea': typea,
        'obesity': obesity,
        'alcohol': alcohol,
        'age': age
    }

    features = pd.DataFrame(data, index=[0])
    return features

# User input features
user_data = user_input_features()

# Predict the outcome
prediction = model.predict(user_data)
prediction_proba = model.predict_proba(user_data)

# Display input data and prediction results
st.subheader('User Input Features')
st.write(user_data)

st.subheader('Prediction')
if prediction[0] == 1:
    st.write('The person is **likely** to have heart disease.')
else:
    st.write('The person is **unlikely** to have heart disease.')

st.subheader('Prediction Probability')
st.write(f"Probability of No Heart Disease: {prediction_proba[0][0]:.2f}")
st.write(f"Probability of Heart Disease: {prediction_proba[0][1]:.2f}")

# Optional: Display Correlation Heatmap
if st.sidebar.checkbox("Show Correlation Heatmap"):
    st.subheader('Correlation Heatmap')
    cols = ['sbp', 'tobacco', 'ldl', 'adiposity', 'famhist', 'typea', 'obesity', 'alcohol', 'age', 'chd']
    cm = np.corrcoef(df[cols].values.T)

    plt.figure(figsize=(15, 10))
    sns.set(font_scale=1.5)
    sns.heatmap(cm, annot=True, square=True, fmt='.2f', yticklabels=cols, xticklabels=cols)
    st.pyplot(plt)
