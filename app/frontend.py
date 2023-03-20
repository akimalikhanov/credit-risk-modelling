import streamlit as st
import config
import requests
from utils import *
import pandas as pd
import json
import os



st.title('Credit Risk Modelling')
st.write('''Predicting probability of default of an applicant using various data sources. 
            You must upload 7 files with exact names as specified in description below.''')
with st.expander("Data sources description."):
        st.write('''
        1. application: the main data with information about each loan application.
        
        2. bureau: data concerning client's previous credits from other financial institutions. Each previous credit has its own row in bureau, but one loan in the application data can have multiple previous credits.
        
        3. bureau_balance: monthly data about the previous credits in bureau. Each row is one month of a previous credit, and a single previous credit can have multiple rows, one for each month of the credit length.
        
        4. previous_app: previous applications for loans of clients who have loans in the application data. Each current loan in the application data can have multiple previous loans. Each previous application has one row and is identified by the feature SK_ID_PREV.
        
        5. cash: monthly data about previous point of sale or cash loans clients have had. Each row is one month of a previous point of sale or cash loan, and a single previous loan can have many rows.
        
        6. card: monthly data about previous credit cards clients have had. Each row is one month of a credit card balance, and a single credit card can have many rows.
        
        7. installments: payment history for previous loans. There is one row for every made payment and one row for every missed payment.
        ''')

test_data=st.file_uploader("All Data", type='csv', accept_multiple_files =True)


cutoff=st.slider('Choose your cutoff value (minimum probability that would be considered a "default", i.e., class-1)', 0., 1., 0.5)
submit=st.button('Submit')
if submit:
    test_dict={i.name: i for i in test_data}
    with st.spinner('Predicting...'):
        if len(test_data)!=7:
            st.error("Please provide all required information!")
        else:
            app_df=pd.read_csv(test_dict['application.csv'])
            del test_dict['application.csv']
            client_id=app_df['SK_ID_CURR'].values[0]
            client_dir=config.CLIENTS_FILE+str(client_id)
            if not os.path.exists(client_dir):
                os.makedirs(client_dir)
            # app_df.to_csv(client_dir+str('/application.csv'), index=False)
            
            file_names=config.CLIENT_FILENAMES

            for fname, d in test_dict.items():
                print(fname, d)
                df=pd.read_csv(d)
                # df.to_csv(os.path.join(client_dir, fname), index=False)

            client_data_dict=dict(zip(config.DF_NAMES, [f'../data/clients/{client_id}/'+dir_ for dir_ in config.CLIENT_FILENAMES]))
            
            data_df=full_sample(client_id, client_data_dict, config.PATH_DICT, config.SAMPLE_DTYPES_DICT)
            data_df.to_csv(os.path.join(client_dir, f'{client_id}_full.csv'), index=False)
            
            data=read_sample(data_df) 
            response=requests.post('http://127.0.0.1:8000/predict', json=data)
            pred=json.loads(response.text)['prediction']
    st.write(f'Probability of Default is {pred:.3f}')
    if pred>cutoff:
        st.error(f'The applicant is likely to default!')
    else:
        st.success(f'The applicant is not likely to default:)')
    
    with st.expander("Prediction Explanation"):
        shap_waterfall(data_df, config.PATH_DICT)
        '''
        This plot show 15 features that affected the prediction the most and in which direction they shifted it.
        \"Blue\" features decrease Probability of Default, while \"red\" ones increase it. Notice that f(x) is a log-odds, not a probability itself.
        The values on the plot are SHAP values. [SHAP](https://christophm.github.io/interpretable-ml-book/shap.html) is a method to interpret predictions of black-box models. 
        Check out [this](https://shap.readthedocs.io/en/latest/index.html) link for more information.
        '''