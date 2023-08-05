from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

app = FastAPI()


# Load the trained k-means model
pickle_in = open("clustering.pkl", "rb")
clustering = joblib.load(pickle_in)

with open('scaling_model.pkl', 'rb') as file:
    loaded_scaling_model = joblib.load(file)

with open('pca.pkl', 'rb') as file:
    loaded_pca_model = joblib.load(file)

with open('one_hot_encoded_df.pkl', 'rb') as file:
    one_hot_encoded_df = joblib.load(file)

@app.get('/{name}')
def get_name(name: str):
    return {'Welcome to the Deployment step': f'{name}'}

class UserInput(BaseModel):
    city: str
    marketing_name:str
    tot_bytes: int
    tot_unknownbytes: int	
    tot_webbytes: int
    tot_mailbytes: int
    tot_chatbytes: int
    tot_voipbytes: int
    DL_THRPUT_ALL: float
    LTE_CONG: int
    date: str
    tot_vpnbytes: int

class FeatureData(BaseModel):
    data: list[UserInput]

def preprocess_data(data: FeatureData):
    input_data = []
    for item in data.data:
        input_data.append({
            'tot_bytes': item.tot_bytes,
            'tot_unknownbytes': item.tot_unknownbytes,
            'tot_webbytes': item.tot_webbytes,
            'tot_mailbytes': item.tot_mailbytes,
            'tot_chatbytes': item.tot_chatbytes,
            'tot_voipbytes': item.tot_voipbytes,
            'tot_vpnbytes': item.tot_vpnbytes,
            'city': item.city,
            'marketing_name': item.marketing_name,
            'DL_THRPUT_ALL': item.DL_THRPUT_ALL,
            'LTE_CONG': item.LTE_CONG,
            'date':item.date
        })

    df = pd.DataFrame(input_data)

    mean_value = df['DL_THRPUT_ALL'].mean()
    df['DL_THRPUT_ALL'] = df['DL_THRPUT_ALL'].fillna(mean_value)
    #print(df.head())

    dat = pd.get_dummies(df, columns=['city', 'marketing_name'])
    for col in one_hot_encoded_df.columns:
       if col not in dat.columns:
            dat[col] = 0
    dat = dat[one_hot_encoded_df.columns]


    for col in df.columns:
        if df[col].dtype == 'uint8':
            df[col] = df[col].astype('int64')
    numeric_data = df.select_dtypes(include=['int64','float64'])
    X = numeric_data

    X_std = loaded_scaling_model.transform(X)
    X_pca=loaded_pca_model.transform(X_std)

    X_pca = pd.DataFrame(X_pca) 
    X_pca.columns = ['P1', 'P2'] 
    X_pca["Cluster"] = clustering.predict(X_pca)
    return X_pca

@app.post('/predict')
def predict(data: FeatureData):
    res = preprocess_data(data)
    clusters = res["Cluster"].values.tolist()

    return {"Clusters": clusters}
