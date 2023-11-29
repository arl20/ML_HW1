from fastapi import FastAPI,  File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Union, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
import re 
import warnings
import math
import pickle
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error as MSE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
import io

app = FastAPI()

class Item(BaseModel):
    name: str
    year: int
    selling_price: Optional[int] = None
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: Union[str, None]
    engine: Union[str, None]
    max_power: Union[str, None]
    torque: Union[str, None]
    seats: Union[float, None]


class Items(BaseModel):
    objects: List[Item]
        
def str_to_float(st):
    if isinstance(st, float) or st is None:
        return st
    num = re.findall('\d+[.,]?\d*', st)
    if len(num) == 0:
        return np.nan
    else:
        return float(num[0])
    
def divide_torque(torque):
    try:
        if not isinstance(torque, str): #case 1
            final_torque, final_max_torque_rmp = torque, torque
        else:
            torque_split = torque.split('@')
            torque_split_at = torque.split('at')
            torque_split_sl = torque.split('/')
            if len(torque_split) == 2 or len(torque_split) == 3 or len(torque_split_at) == 2 or len(torque_split_sl) == 2: #cases 2, 3, 4, 6
                if len(torque_split) == 2:
                    left, right = torque_split
                elif len(torque_split) == 3:
                    left, right = torque_split[0],  torque_split[1]
                elif len(torque_split_at) == 2:
                    left, right = torque_split_at
                else:
                    left, right = torque_split_sl
                if '+/-' in right:
                    num, interval = re.findall('\d+[.,]?\d*', right)
                    if interval.find(',' ) != -1:
                        interval = interval.replace(',', '')
                    if num.find(',' ) != -1:
                        num = num.replace(',', '')
                    final_max_torque_rmp = float(num) + float(interval) 
                else:
                    num = re.findall('\d+[.,]?\d*', right)[-1]
                    if num.find(',' ) != -1:
                        num = num.replace(',', '')
                    final_max_torque_rmp = float(num)
                num = re.findall('\d+[.,]?\d*', left)[0]
                if num.find(',' ) != -1:
                    num = num.replace(',', '.')
                final_torque = float(num)
            elif len(torque_split) == 1 or len(torque_split) == 1 or len(torque_split_at) == 1 or len(torque_split_sl) == 1: # case 5
                num = re.findall('\d+[.,]?\d*', torque)[0]
                if num.find(',' ) != -1:
                    num = num.replace(',', '.')
                final_torque, final_max_torque_rmp = float(num), np.nan
            else:
                print('!', torque_split, torque_split_at, torque_split_sl)
                final_torque, final_max_torque_rmp = np.nan, np.nan
                warnings.warn(f'Unexpected line structure: {torque}. Returned empty value')
    except:
        final_torque, final_max_torque_rmp = np.nan, np.nan
        warnings.warn(f'Unexpected line structure: {torque}. Returned empty value')
    finally:
        return final_torque, final_max_torque_rmp 

    
def prepare_data_and_get_predict(test):
    df_test = test.copy()
    for col in ('mileage', 'engine', 'max_power'):
        df_test[f'{col}'] = df_test[col].apply(str_to_float)
    df_test['tmp_torque'] = df_test['torque'].apply(divide_torque)
    df_test['torque'] = df_test['tmp_torque'].apply(lambda x: x[0])
    df_test['max_torque_rpm'] = df_test['tmp_torque'].apply(lambda x: x[1])
    df_test = df_test.drop('tmp_torque', axis=1)
    with open('final_model.pkl', 'rb') as file:
        model_loaded = pickle.load(file)
    for column in ('mileage', 'engine', 'max_power', 'torque' , 'max_torque_rpm', 'seats'):
        df_test[column] = df_test[column].fillna(model_loaded['medians_train'][column])
    df_test['max_torque_rpm_log'] = df_test['max_torque_rpm'].apply(math.log) 
    df_test['km_driven_log'] = df_test['km_driven'].apply(math.log) 
    df_test['model'] = df_test['name'].apply(lambda x: x.split()[0])
    df_test['year_2'] = df_test['year'] ** 2
    df_test['engine_log'] = df_test['engine'].astype(int).apply(math.log) 
    X_test_cat = df_test.drop(['selling_price', 'name', 'year', 'max_torque_rpm', 'km_driven', 'engine'], axis=1)
    X_test_cat['seats'] = X_test_cat['seats'].astype(int).astype(str)
    X_test_cat = pd.get_dummies(X_test_cat, drop_first = True)
    for col in set(model_loaded['train_columns']) - set(X_test_cat):
        X_test_cat[col] = 0
    X_test_cat_norm = model_loaded['scaler'].transform(X_test_cat[list(model_loaded['train_columns'])])
    y_pred_test = list(map(math.exp, model_loaded['model'].predict(X_test_cat_norm)))
    return y_pred_test
    
@app.post("/predict_item")
def predict_item(item: Item) -> float:
    df_test = pd.DataFrame([dict(item)])
    y_pred_test = prepare_data_and_get_predict(df_test)
    return y_pred_test[0]


@app.post("/predict_items", response_class=StreamingResponse)
def predict_items(file: UploadFile = File(...)):
    df_test = pd.read_csv(file.file)
    y_pred_test = prepare_data_and_get_predict(df_test)
    stream = io.StringIO()
    df_test['prediction'] = y_pred_test
    df_test.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]),
                                 media_type="text/csv")
    return response
