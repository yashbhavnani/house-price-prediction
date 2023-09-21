#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pickle
import streamlit as st


# In[2]:


# loading the saved model
loaded_model = pickle.load(open('house_price_predictor.sav', 'rb'))


# In[4]:


def price_prediction(input_data):
    # changing the input_data to a numpy array
    input_data_array = np.asarray(input_data)

    # reshaping the array as we are predicting only for one instance
    input_data_reshaped = input_data_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)

    return 'The predicted price of the house is: ',prediction.tolist()


# In[7]:


def main():
    
    # giving a title
    st.title('House Price Prediction Web App')
    
    # getting the input data from the user
    CRIM = st.text_input('Per capita Crime Rate by Town: ')
    ZN = st.text_input('Proportion of residential land zoned for lots over 25,000 sq.ft: ')
    INDUS = st.text_input('Proportion of non-retail business acres per town: ')
    CHAS = st.text_input('Charles River dummy variable (= 1 if tract bounds river; 0 otherwise ): ')
    NOX = st.text_input('Nitric Oxides concentration (parts per 10 million): ')
    RM = st.text_input('Average number of rooms per dwelling: ')
    AGE = st.text_input('Proportion of owner-occupied units built prior to 1940: ')
    DIS = st.text_input('Weighted distances to five Boston employment centres: ')
    RAD = st.text_input('Index of accessibility to radial highways: ')
    TAX = st.text_input('Full value property tax rate per 10,000$: ')
    PTRATIO = st.text_input('Pupil-Teacher ratio by town: ')
    B = st.text_input('1000(Bk-0.63)^2 where Bk is the proportion of blacks by town: ')
    LSTAT = st.text_input('% lower status of the population: ')
    
    # code for prediction
    result = ''
    
    # creating a button for prediction
    if st.button('Predict'):
        result = price_prediction([CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT])
        
    st.success(result)
    


# In[8]:


if __name__ == '__main__':
    main()


# In[ ]:




