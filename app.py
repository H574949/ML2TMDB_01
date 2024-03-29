#!/usr/bin/env python
# coding: utf-8

# In[2]:


from pycaret.classification import load_model, predict_model
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
import os


# In[ ]:


class StreamlitApp:
    
    def __init__(self):
        self.model = load_model('revenue_pred')
        self.save_fn = 'path.csv'

    def predict(self, input_data):
        return predict_model(self.model, data=input_data)

    def store_prediction(self, output_df):
        if os.path.exists(self.save_fn):
            safe_df = pd.read_csv(self.save_fn)
            safe_df = safe_df.append(output_df, ignore_index=True)
            safe_df.to_csv(self.save_fn, index=False)

        else:
            output_df.to_csv(self.save_fn, index=False)
    
    def run(self):
        image = Image.open('./image.png')
        st.image(image, use_column_width=False)

        
        st.sidebar.info('This app is created to predict the revenue of a film')
        st.sidebar.success('Success!')
        st.title('Film revenue prediction')

        if True:
            budget = st.number_input('Budget', min_value=10000, max_value=10000000, value=10000)
            popularity = st.number_input('TMBD Popularity Rating', min_value=0, max_value=100, value=0)
            runtime = st.number_input('Runtime', min_value=1, max_value=600, value=1)
            genre = st.selectbox('Genres', ['Comedy','Action','Thriller','Drama'])
            BTC = 0
            production_companies = st.text_input("Production company")
            original_language = st.selectbox('Original Language', ['en', 'hi','ko','sr','fr','it','nl','zh','es','cs','ta','cn','ru','tr','ja','fa','sv','de','te','pt','mr'])
            date = st.date_input("Release date")
            if st.checkbox('Belongs to collection'):
                BTC = 1



            output =''
            input_dict = {'budget':budget, 'popularity':popularity, 'runtime':runtime, 'genres':genre, 'production_companies':production_companies,
                        'original_language':original_language, 'release_date':date}
            input_df = pd.DataFrame(input_dict, index=[0])

            if st.button('Predict'):
                output = self.predict(input_df)
                self.store_prediction(output)
                output = (output['Label'][0])
                #'test' if output['Label'][0] == 1 else 'not'
                
            st.success('Predicted Revenue: {}'.format(output))


sa = StreamlitApp()
sa.run()
