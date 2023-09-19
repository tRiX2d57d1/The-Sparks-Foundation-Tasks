# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 00:40:04 2023

@author: Asus
"""

import numpy as np
import pickle
import streamlit as st

loaded_model1 = pickle.load(open('C:/Users/Asus/Documents/TSF INTERNSHIP/score_prediction.sav','rb'))

def score_prediction(input_data):
    

    input_data_arr = np.asarray(input_data,dtype=object)

    reshaped = input_data_arr.reshape(1,-1)

    prediction = loaded_model1.predict(reshaped)
    return prediction


def main():
    st.title('What will be the score of the student if he studied for 9.25 hrs/day?')
    
    hours = st.text_input('Hours')
    pred = ' '
    
    if st.button('Predict Score'):
        pred = score_prediction(hours)
    
    st.success(pred)
    


if __name__ == '__main__':
    main()