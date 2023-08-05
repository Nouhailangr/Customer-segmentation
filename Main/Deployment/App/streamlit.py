import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load the trained k-means model
pickle_in = open("clustering.pkl", "rb")
clustering = joblib.load(pickle_in)

with open('scaling_model.pkl', 'rb') as file:
    loaded_scaling_model = joblib.load(file)

with open('pca.pkl', 'rb') as file:
    loaded_pca_model = joblib.load(file)

with open('one_hot_encoded_df.pkl', 'rb') as file:
    one_hot_encoded_df = joblib.load(file)


def preprocess_data(input_data):

    df = pd.DataFrame(input_data)

    # Handle missing values
    mean_value = df['DL_THRPUT_ALL'].mean()
    df['DL_THRPUT_ALL'] = df['DL_THRPUT_ALL'].fillna(mean_value)

    # Handle missing values for other numeric columns as well if required
    # ...

    # One-hot encode categorical features
    dat = pd.get_dummies(df, columns=['city', 'marketing_name'])
    for col in one_hot_encoded_df.columns:
        if col not in dat.columns:
            dat[col] = 0
    dat = dat[one_hot_encoded_df.columns]

    # Convert uint8 columns to int64
    for col in df.columns:
        if df[col].dtype == 'uint8':
            df[col] = df[col].astype('int64')

    # Select numeric features for prediction
    numeric_data = df.select_dtypes(include=['int64', 'float64'])
    X = numeric_data

    # Scale the features using the loaded scaling model
    X_std = loaded_scaling_model.transform(X)

    # Perform PCA transformation using the loaded PCA model
    X_pca = loaded_pca_model.transform(X_std)

    X_pca = pd.DataFrame(X_pca)
    X_pca.columns = ['P1', 'P2']
    X_pca["Cluster"] = clustering.predict(X_pca)
    return X_pca


def predict(input_data):
    res = preprocess_data(input_data)
    clusters = res["Cluster"].values.tolist()

    return {"Clusters": clusters}


def main():
    st.title('Customer segmentation Web App')
    city = st.text_input('City')
    marketing_name = st.text_input('Marketing name')
    tot_bytes = st.text_input('Total Bytes')
    tot_unknownbytes = st.text_input('Total unknown bytes')
    tot_webbytes = st.text_input('Total web bytes')
    tot_mailbytes = st.text_input('Total mail bytes')
    tot_chatbytes = st.text_input('Total chat bytes')
    tot_voipbytes = st.text_input('Total voip bytes')
    tot_vpnbytes = st.text_input('Total VPN bytes')
    DL_THRPUT_ALL = st.text_input('Debit throughput')
    LTE_CONG = st.text_input('LTE Congestion')
    date = st.text_input('Date')

    # Code for Prediction
    clust = ''

    # Creating a button for Prediction
    if st.button('Customer Segmentation'):
        input_data = {
            'city': city,
            'marketing_name': marketing_name,
            'tot_bytes': float(tot_bytes),
            'tot_unknownbytes': float(tot_unknownbytes),
            'tot_webbytes': float(tot_webbytes),
            'tot_mailbytes': float(tot_mailbytes),
            'tot_chatbytes': float(tot_chatbytes),
            'tot_voipbytes': float(tot_voipbytes),
            'tot_vpnbytes': float(tot_vpnbytes),
            'DL_THRPUT_ALL': float(DL_THRPUT_ALL),
            'LTE_CONG': float(LTE_CONG),
            'date': date
        }

        result = predict([input_data])
        clust = result["Clusters"][0]
        

    st.success(f'Predicted Cluster: {clust}')
    


if __name__ == '__main__':
    main()
