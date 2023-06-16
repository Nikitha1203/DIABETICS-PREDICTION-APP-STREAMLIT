# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 13:52:10 2023

@author: My PC
"""
import numpy as np
import pickle
import streamlit as st


loaded_model = pickle.load(open('C:/Users/My PC/Desktop/STRAML/trained_model.sav','rb'))


# Function creation
def daibetics_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if prediction[0] == 0:
        return "Yes, the person is healthy."
    else:
        return "Yes, the person is suffering from diabetes."


def main():
    st.title("DAIBETICS PREDICTION APP")

    

    # Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
    Pregnancies = st.text_input("Number of Pregnancies")
    Glucose = st.text_input("Glucose level")
    BloodPressure = st.text_input("Blood Pressure value")
    SkinThickness = st.text_input("Skin Thickness value")
    Insulin = st.text_input("Insulin Level")
    BMI = st.text_input("BMI Value")
    DiabetesPedigreeFunction = st.text_input("Prediction function")
    Age = st.text_input("Age of a Person")

    diagnosis = ''

    # Button
    if st.button('DIABETICS RESULT'):
        diagnosis = daibetics_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        st.success(diagnosis)

if __name__ == '__main__':
    main()