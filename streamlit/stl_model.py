import streamlit as st     
import pandas as pd 
from model.train.train import train
from model.test.test import test

def train_mode():

    st.subheader('Parameter')
    epoch_col, batch_size_col, earling_stop_col = st.columns(3)

    with epoch_col:
        epochs = st.number_input(
            "Number of epochs",
            min_value=0,
            value=0,
            step=1,
            key='epoch_box'
        )
    with batch_size_col:
        batch_size = st.number_input(
            "Number of batch size",
            min_value=0,
            value=0,
            step=1,
            key='batch_size_box'
        )
    with earling_stop_col:
        early_stop = st.number_input(
            "Early stopping",
            min_value=0,
            value=0,
            step=1,
            key='earling_stop_box'
        )

    start_training = st.button('Start training', key='train_start_button')

    if start_training:
        with st.spinner('Wait for it...'):
            train(epochs=epochs, batch_size=batch_size, patience_of_early_stopping=early_stop)
        st.write("Done")

def test_mode():
    st.subheader('Parameter')
    province_col, feature_col = st.columns(2)

    with province_col:
        province = st.selectbox(
            "Choice province",
            st.session_state.provinces,
        )
    with feature_col:
        feature = st.selectbox(
            "Choice feature",
            st.session_state.numerical_columns
        )   
    
    start_evaluation = st.button('Start evaluation', key='evaluation_start_button')

    if start_evaluation:
        y_df, predict_df = test(province=province)

        y = y_df[[feature]]
        y.columns = ['True value']

        predict = predict_df[[feature]]
        predict.columns = ['Predict']

        st.line_chart(pd.concat([y, predict], axis=1))

