import streamlit as st
import pandas as pd
from dataset.clean_data.preprocessing_data import preprocessing
from dataset.train_val_data.generate_data import generate_data

def data_preparation_mode():
    st.header('Data preparation')

    # Preprocessing
    st.divider()
    st.subheader('Preprocessing')
    # select scale feature 
    scaled_features = st.multiselect(
        "1\. Feature scaling",
        ['All'] + st.session_state.numerical_columns,
        key='feature_scaling_box'
    )  
    if 'All' in scaled_features:
        scaled_features = st.session_state.numerical_columns

    # Select encode feature
    encoded_features = st.multiselect(
        '2\. Feature encoding',
        ['All'] + st.session_state.nominal_columns,
        key='feature_encoding_box'
    )    
    if 'All' in scaled_features:
        encoded_features = st.session_state.nominal_columns
    
    # Result
    if 'clean_df' not in st.session_state:
        st.session_state.clean_df = pd.DataFrame()
    if st.button('Start preprocessing'):
        clean_df = preprocessing(scale_columns=scaled_features, encode_columns=encoded_features)
        st.session_state.clean_df = clean_df
    if 'clean_df' in st.session_state:
        if st.session_state.clean_df.empty == False:
            st.dataframe(st.session_state.clean_df)

    # Initialize parameter
    st.divider()
    st.subheader('Data generation')
    
    # Define
    if 'split_ratio' not in st.session_state:
        st.session_state.split_ratio = 0
    if 'future_length' not in st.session_state:
        st.session_state.future_length = 0
    if 'past_length' not in st.session_state:
        st.session_state.past_length = 0
    if 'step' not in st.session_state:
        st.session_state.step = 0
    if 'target_feature' not in st.session_state:
        st.session_state.target_feature = []
    # Split ratio
    split_ratio = st.slider(
        '1\. Split ratio', 
        min_value=0.0, 
        max_value=1.0, 
        value=0.0, 
        step=0.1, 
        key='split_ratio_slider')

    # Day parameter
    future_col, past_col, step_col = st.columns(3)
    with future_col:
        st.session_state.future_lenght = st.number_input(
            '2\. Number of predicted days',
            min_value=0,
            step=1,
            key='future_length_input'
            )
    with past_col:
        st.session_state.past_length = st.number_input(
            '3\. Number of past days',
            min_value=0,
            step=1,
            key='past_length_input'
            )
    with step_col:
        st.session_state.step = st.number_input(
            '4\. Step',
            min_value=0,
            step=1,
            key='step_input'
            )
        
    # Target feature 
    if st.session_state.clean_df.empty:
        target_features = []
    else:
        target_features = ['All']+list(st.session_state.clean_df.columns)

    st.session_state.target_feature = st.multiselect(
        '5\. Target feature (Preprocessing first)',
        target_features,
        key='target_feature_input'
    )

    # Generrate button
    start_generate = st.button('Start generating', key='generate_start_button')
    if start_generate:
        with st.spinner('Wait for it...'):
            generate_data()