#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pickle-mixin


# In[2]:


import numpy as np
import pickle


# In[3]:


# loading the saved model
loaded_model = pickle.load(open('house_price_predictor.sav', 'rb'))


# In[16]:


# input_data = (0.15682292, -0.4898311 ,  0.98336806, -0.27288841,  0.47919371,
#         10.28867984,  0.87020968, -0.68730678,  1.63579367,  1.50571521,
#         0.81196637,  0.44624347,  10.81480158)
def price_prediction(input_data):
    # changing the input_data to a numpy array
    input_data_array = np.asarray(input_data)

    # reshaping the array as we are predicting only for one instance
    input_data_reshaped = input_data_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)

    return 'the predicted price is :',prediction.tolist()

# features = np.array([[ 0.15682292, -0.4898311 ,  0.98336806, -0.27288841,  0.47919371,
#         10.28867984,  0.87020968, -0.68730678,  1.63579367,  1.50571521,
#         0.81196637,  0.44624347,  10.81480158]])
# prediction = loaded_model.predict(features)
# print(prediction)


# In[18]:


result = price_prediction((0.15682292, -0.4898311 ,  0.98336806, -0.27288841,  0.47919371,
        10.28867984,  0.87020968, -0.68730678,  1.63579367,  1.50571521,
        0.81196637,  0.44624347,  10.81480158))


# In[19]:


print(result)


# In[ ]:




