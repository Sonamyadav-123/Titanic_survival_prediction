import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the pre-trained model
model = joblib.load('knn_Titanic.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load('columns.pkl')

st.title('🚢 Titanic Survival Prediction App ⚓')
st.markdown("⬇️proveide the following details to predict if a passenger would survive the Titanic disaster.")

Pclass= st.slider("Pclass",1,3,1)
Sex_input= st.selectbox("sex",['male','female'])
Age= st.slider("Age",0,100,20)
SibSp= st.slider("SibSp",0,10,3) 
Parch= st.slider("Parch",0,10,3) 
Fare = st.number_input("Fare", min_value=0.0, value=32.0)
Embarked_input = st.selectbox("Embarked", ['C', 'Q', 'S'])
Sex = 1 if Sex_input == 'male' else 0
Embarked = 0 if Embarked_input == 'C' else (1 if Embarked_input == 'Q' else 2)


if st.button("predict"):
    raw_data = {
        'Pclass': Pclass,
        'Sex': Sex,
        'Age': Age,
        'SibSp': SibSp,
        'Parch': Parch,
        'Fare': Fare,
        'Embarked': Embarked
    }

    input_data = pd.DataFrame([raw_data])
    

    for col in expected_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    input_df = input_data[expected_columns]

    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]
    
    if prediction == 0:
        st.error("❌ The passenger would not survive.")
    else:
        st.success("✅ The passenger would survive.")