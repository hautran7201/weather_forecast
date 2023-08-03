import streamlit as st
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.insert(5, './')

from stl_dataset import dataset_mode
from stl_data_preparation import data_preparation_mode
from stl_model import train_mode


# Stream lit
if 'dataset' not in st.session_state:
    st.session_state.dataset = pd.read_csv(r'dataset\raw_data\weather.csv').drop('Unnamed: 0', axis=1)
    st.session_state.provinces = st.session_state.dataset['province'].unique()
    st.session_state.numerical_columns = st.session_state.dataset.select_dtypes(include=['number']).columns.tolist()
    st.session_state.nominal_columns = st.session_state.dataset.select_dtypes(exclude=['number']).columns.tolist()


# Side bar
with st.sidebar:
    st.markdown('# **Weather forecast**')
    option = st.selectbox(
        'Select option',
        ['Dataset', 'Data preparation', 'Model'],
        key='select_option'
    )

# Option
if option == 'Dataset':
    dataset_mode()
elif option == 'Data preparation':
    data_preparation_mode()
else:
    train_mode()
    


