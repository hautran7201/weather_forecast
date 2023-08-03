import streamlit as st
import missingno as msno
import pandas as pd
from PIL import Image

def dataset_mode():
    # Dataset header
    st.header('Dataset')
    province_box, feature_box = st.columns(2)

    # Selected province box
    with province_box:
        province = st.selectbox(
            "Province",
            list(st.session_state.provinces),
            key='province_dataset_box'
        )

    # Selected feature box
    with feature_box:
        features = st.multiselect(
            'Feature',
            ['All']+list(st.session_state.dataset.columns),
            ['All'],
            key='feature_dataset_box'
        )
    if 'All' in features or [] == features:
        filter_df = st.session_state.dataset[st.session_state.dataset['province']==province]
    else:
        filter_df = st.session_state.dataset[st.session_state.dataset['province']==province][features]
    st.dataframe(filter_df)


    # Statistic header
    st.divider()
    st.subheader('Statistic')

    st.markdown('1. Describe')
    describe = st.session_state.dataset.describe().round(2)
    st.dataframe(describe)

    st.markdown('2. Missing value')
    fig = msno.matrix(st.session_state.dataset)
    fig_copy = fig.get_figure()
    fig_copy.savefig(r'dataset\raw_data\missing_data.png')
    st.image(Image.open(r'dataset\raw_data\missing_data.png'), 'Missing value')

    
    # Statistic header
    st.divider()
    st.subheader('Visualize')

    st.markdown('1. Plot numerical feature')
    # Selected province box
    province_box, feature_box = st.columns(2)
    with province_box:
        province = st.selectbox(
            "Province",
            list(st.session_state.provinces),
            key='province_scatterplot_box'
        )
    # Selected feature box
    with feature_box:
        features = st.selectbox(
            'Feature',
            st.session_state.numerical_columns,
            key='feature_scatterplot_box'
        )
    filter_df = st.session_state.dataset[st.session_state.dataset['province']==province]
    st.line_chart(filter_df, x='date', y=features)

    st.markdown('2. Plot nominal feature')
    # Selected province box
    province_box, feature_box = st.columns(2)
    with province_box:
        province = st.selectbox(
            "Province",
            list(st.session_state.provinces),
            key='province_barplot_box'
        )
    value_counts = st.session_state.dataset[st.session_state.dataset['province']==province]['wind_d'].value_counts()
    df_val_counts = pd.DataFrame(value_counts)
    df_value_counts_reset = df_val_counts.reset_index()
    df_value_counts_reset.columns = ['wind_d', 'counts'] # change column names
    st.bar_chart(df_value_counts_reset, x='wind_d', y='counts')