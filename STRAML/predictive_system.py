# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 13:36:20 2023

@author: My PC
"""
import numpy as np 
import pickle
loaded_model = pickle.load(open('C:/Users/My PC/Desktop/STRAML/trained_model.sav','rb'))
input_data = (5,166,72,19,175,25.8,0.587,51)
input_data_as_numpy_array = np.asarray(input_data)
#reshaping data 
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
    print("yes person is healthy")
else:
    print("yes person suffering from daibetics")