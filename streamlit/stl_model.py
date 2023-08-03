import streamlit as st     
from model.train.train import train

def train_mode():
    st.header("Model")
    train_mode, valuation_mode, inference_mode = st.tabs(["Train", "Valuation", "inference"])

    with train_mode:
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
            batch_size = st.number_input(
                "Early stopping",
                min_value=0,
                value=0,
                step=1,
                key='earling_stop_box'
            )

        start_training = st.button('Start training', key='train_start_button')

        if start_training:
            with st.spinner('Wait for it...'):
                train()
            st.write("Done")