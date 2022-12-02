from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union
import pandas as pd
import numpy as np
import random
import re
import datetime
import sklearn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import pickle

random.seed(42)
np.random.seed(42)




app = FastAPI()


class Item(BaseModel):
    name: str
    year: Union[int, None] = None
    selling_price: int
    km_driven: Union[int, None] = None
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: Union[str, None] = None
    engine: Union[str, None] = None
    max_power: Union[str, None] = None
    torque: Union[str, None] = None
    seats: Union[float, None] = None




class Items(BaseModel):
    objects: List[Item]


@app.post("/test")
def predict_item(item: Item): # -> float
    if item.year == None:
        item.year = 123
    return item.year


@app.post("/predict_item")
def predict_item(item: Item): # -> float

    # --------------------------------------------------------------------------
    # обработка признаков
    try:
        num = item.mileage.split()
        item.mileage = float(num[0])
    except:
        item.mileage = None

    try:
        num = item.engine.split()
        item.engine = float(num[0])
    except:
        item.engine = None

    try:
        num = item.max_power.split()
        item.max_power = float(num[0])
    except:
        item.max_power = None

    torque = []
    max_torque_rpm = []
    if (item.torque != None):
        cond0 = re.findall(r'\+', item.torque)
        if cond0 == []:
            nums = re.findall(r'\d+[.,\']\d+|\d+', item.torque)
            cond1 = re.findall(r'kgm|KGM|Kgm', item.torque)
            cond2 = re.findall(r'Nm|nm|NM', item.torque)
            cond3 = re.findall(r'rpm|RPM|Rpm', item.torque)
            cond_len = len(nums)
            if (cond2 != []) and (cond3 != []) and (cond_len == 2):
                torque.append(nums[0])
                max_torque_rpm.append(nums[1])
            elif (cond2 != []) and (cond3 != []) and (cond_len == 3):
                torque.append(nums[0])
                max_torque_rpm.append(nums[2])
            elif (cond1 != []) and (cond3 != []) and (cond_len == 2):
                torque.append(float(nums[0]) * 9.8)
                max_torque_rpm.append(nums[1])
            elif (cond1 != []) and (cond3 != []) and (cond_len == 3):
                torque.append(float(nums[0]) * 9.8)
                max_torque_rpm.append(nums[2])

            elif (cond2 == []) and (cond3 != []) and (cond1 == []) and (cond_len == 1):
                torque.append(None)
                max_torque_rpm.append(nums[0])
            elif (cond2 == []) and (cond3 != []) and (cond1 == []) and (cond_len == 2):
                torque.append(None)
                max_torque_rpm.append(nums[1])
            elif (cond2 == []) and (cond3 != []) and (cond1 == []) and (cond_len == 3):
                torque.append(nums[0])
                max_torque_rpm.append(nums[2])

            elif (cond2 != []) and (cond3 == []) and (cond1 == []) and (cond_len == 1):
                torque.append(nums[0])
                max_torque_rpm.append(None)
            elif (cond2 != []) and (cond3 == []) and (cond1 == []) and (cond_len == 2):
                torque.append(nums[1])
                max_torque_rpm.append(None)

            elif (cond2 == []) and (cond3 == []) and (cond1 != []) and (cond_len == 1):
                torque.append(float(nums[0]) * 9.8)
                max_torque_rpm.append(None)
            elif (cond1 == []) and (cond3 == []) and (cond1 != []) and (cond_len == 2):
                torque.append(float(nums[1]) * 9.8)
                max_torque_rpm.append(None)

            elif (cond1 == []) and (cond3 == []) and (cond1 == []) and (cond_len == 2):
                torque.append(nums[0])
                max_torque_rpm.append(nums[1])
            elif (cond1 == []) and (cond3 == []) and (cond1 == []) and (cond_len == 3):
                torque.append(nums[0])
                max_torque_rpm.append(nums[2])
        elif cond0 == ['+']:
            nums = re.findall(r'\d+[.,\']\d+|\d+', item.torque)
            cond1 = re.findall(r'kgm|KGM|Kgm', item.torque)
            cond2 = re.findall(r'Nm|nm|NM', item.torque)
            cond3 = re.findall(r'rpm|RPM|Rpm', item.torque)
            cond_len = len(nums)
            if (cond3 != []) and (cond2 != []) and (cond_len == 3):
                cond4 = re.findall(r',', item.torque)
                if cond4 != []:
                    torque.append(nums[0])
                    nums2 = nums[1].split(',')
                    nums3 = float(nums2[0] + nums2[1])
                    max_torque_rpm.append(nums3 + float(nums[2]))
                elif cond4 == []:
                    torque.append(nums[0])
                    max_torque_rpm.append(float(nums[1]) + float(nums[2]))
    elif (item.torque == None):
        torque.append(None)
        max_torque_rpm.append(None)

    index4 = []
    for i in range(len(max_torque_rpm)):
        if (str(max_torque_rpm[i]) != 'None') and ((type(max_torque_rpm[i]) == str)):
            cond5 = re.findall(r',', max_torque_rpm[i])
            if cond5 != []:
                nums2 = max_torque_rpm[i].split(',')
                max_torque_rpm[i] = float(nums2[0] + nums2[1])
        index4.append(i)

    item.torque = torque[0]
    df_train = pd.DataFrame([item.dict()])
    df_train['max_torque_rpm'] = max_torque_rpm[0]

    # --------------------------------------------------------------------------
    # Обработка пропусков
    medians = pd.read_csv('medians.csv')
    for col in medians.columns:
        for i in range(len(df_train)):
            if df_train[col][i] == None:
                df_train.loc[i, col] = medians.loc[0, col]


    # --------------------------------------------------------------------------
    df_train['engine'] = df_train['engine'].astype(int)
    df_train['seats'] = df_train['seats'].astype(int)
    df_train['torque'] = df_train['torque'].astype(float)
    df_train['max_torque_rpm'] = df_train['max_torque_rpm'].astype(float)


    # feature engineering
    df_train['year_sq'] = df_train['year'] ** 2
    age_of_car = (datetime.datetime.now().year - df_train['year']) + 1
    df_train['km_driven_per_year'] = df_train['km_driven'] / age_of_car
    df_train['specific power'] = df_train['max_power'] / df_train['engine']
    company = []
    for i in range(len(df_train)):
        comp = df_train['name'][i].split()[0]
        company.append(comp)

    df_train['company'] = company
    list_comp = ['Mitsubishi', 'Volvo', 'Jaguar', 'Force', 'Isuzu', 'Land', 'MG', 'Daewoo', 'Kia', 'Ambassador',
                 'Lexus', 'Peugeot']
    for i in range(len(df_train)):
        if df_train['company'][i] in list_comp:
            df_train.loc[i, 'company'] = 'other'
    owner_2 = []
    for i in range(len(df_train)):
        if (df_train['owner'][i] == 'First Owner') or (df_train['owner'][i] == 'Test Drive Car'):
            owner_2.append('First Owner & Test Drive Car')
        else:
            owner_2.append('Second & Above Owner')

    df_train['owner_2'] = owner_2


    # --------------------------------------------------------------------------
    # подготовка данных к построению модели
    X_cat = df_train[['fuel', 'seller_type', 'transmission', 'owner', 'company', 'owner_2', 'seats']]
    X_real = df_train[['year', 'km_driven', 'mileage', 'engine', 'max_power', 'torque',
                       'max_torque_rpm', 'year_sq', 'km_driven_per_year', 'specific power']]
    y_train = df_train['selling_price']

    #one_hot_enc = OneHotEncoder(drop='first', handle_unknown='ignore')
    #one_hot_enc.fit(X_cat)
    pkl_filename = "pickle_encoder.pkl"
    with open(pkl_filename, 'rb') as file:
        one_hot_enc = pickle.load(file)
    names_of_cat = []
    for i in range(len(one_hot_enc.categories_)):
        for j in range(1, len(one_hot_enc.categories_[i])):
            names_of_cat.append(one_hot_enc.categories_[i][j])

    X_dum01 = one_hot_enc.transform(X_cat)
    X_dum_train = pd.DataFrame(X_dum01.toarray(), columns=names_of_cat)
    X_dum = pd.concat([X_real, X_dum_train], axis=1)
    # X_num = X_dum.columns[:10]
    # X_cat = X_dum.columns[10:]
    # X_dum_v01 = X_dum[X_num]
    # X_cat_v01 = X_dum[X_cat]
    #
    # # стандартизируем вещественные признаки
    # #scaler = StandardScaler()
    # #scaler.fit(X_dum_v01)
    # pkl_filename = "pickle_scaler.pkl"
    # with open(pkl_filename, 'rb') as file:
    #     scaler = pickle.load(file)
    # X_real_stand = scaler.transform(X_dum_v01)
    # X_num_real_stand = pd.DataFrame(data=X_real_stand, columns=X_dum_v01.columns)
    #
    # X_dum_01 = pd.concat([X_num_real_stand, X_cat_v01], axis=1)

    # --------------------------------------------------------------------------
    # построение модели
    # model_ridge_1 = Ridge(alpha=1)
    # model_ridge = model_ridge_1.fit(X_dum_01, y_train)
    pkl_filename = "pickle_model_v02.pkl"
    with open(pkl_filename, 'rb') as file:
        model_ridge = pickle.load(file)

    y_pred_ridge = model_ridge.predict(X_dum)



    #train_r2 = r2_score(y_train, y_pred_ridge)
    pd.set_option('display.max_columns', None)
    #a = str(X_real.iloc[0])
    a = str(df_train)

    # выводить str или int
    return y_pred_ridge[0]





@app.post("/predict_items")
def predict_items(items: List[Item]):   # -> List[float]
    df_train = pd.DataFrame([items[0].dict()])
    for i in range(1, len(items)):
        string = pd.DataFrame([items[i].dict()])
        df_train = pd.concat([df_train, string])

    df_train.reset_index(inplace=True, drop=True)

    # --------------------------------------------------------------------------
    # обработка признаков
    for i in range(len(df_train)):
        try:
            num = df_train.loc[i, 'mileage'].split()
            df_train.loc[i, 'mileage'] = float(num[0])
        except:
            df_train.loc[i, 'mileage'] = None

    for i in range(len(df_train)):
        try:
            num = df_train.loc[i, 'engine'].split()
            df_train.loc[i, 'engine'] = float(num[0])
        except:
            df_train.loc[i, 'engine'] = None

    for i in range(len(df_train)):
        try:
            num = df_train.loc[i, 'max_power'].split()
            df_train.loc[i, 'max_power'] = float(num[0])
        except:
            df_train.loc[i, 'max_power'] = None

    df_train['mileage'] = df_train['mileage'].astype(float)
    df_train['engine'] = df_train['engine'].astype(float)
    df_train['max_power'] = df_train['max_power'].astype(float)

    torque = []
    max_torque_rpm = []
    for i in range(len(df_train['torque'])):
        if (df_train['torque'][i] != None):
            cond0 = re.findall(r'\+', df_train['torque'][i])
            if cond0 == []:
                nums = re.findall(r'\d+[.,\']\d+|\d+', df_train['torque'][i])
                cond1 = re.findall(r'kgm|KGM|Kgm', df_train['torque'][i])
                cond2 = re.findall(r'Nm|nm|NM', df_train['torque'][i])
                cond3 = re.findall(r'rpm|RPM|Rpm', df_train['torque'][i])
                cond_len = len(nums)
                if (cond2 != []) and (cond3 != []) and (cond_len == 2):
                    torque.append(nums[0])
                    max_torque_rpm.append(nums[1])
                elif (cond2 != []) and (cond3 != []) and (cond_len == 3):
                    torque.append(nums[0])
                    max_torque_rpm.append(nums[2])
                elif (cond1 != []) and (cond3 != []) and (cond_len == 2):
                    torque.append(float(nums[0]) * 9.8)
                    max_torque_rpm.append(nums[1])
                elif (cond1 != []) and (cond3 != []) and (cond_len == 3):
                    torque.append(float(nums[0]) * 9.8)
                    max_torque_rpm.append(nums[2])

                elif (cond2 == []) and (cond3 != []) and (cond1 == []) and (cond_len == 1):
                    torque.append(None)
                    max_torque_rpm.append(nums[0])
                elif (cond2 == []) and (cond3 != []) and (cond1 == []) and (cond_len == 2):
                    torque.append(None)
                    max_torque_rpm.append(nums[1])
                elif (cond2 == []) and (cond3 != []) and (cond1 == []) and (cond_len == 3):
                    torque.append(nums[0])
                    max_torque_rpm.append(nums[2])

                elif (cond2 != []) and (cond3 == []) and (cond1 == []) and (cond_len == 1):
                    torque.append(nums[0])
                    max_torque_rpm.append(None)
                elif (cond2 != []) and (cond3 == []) and (cond1 == []) and (cond_len == 2):
                    torque.append(nums[1])
                    max_torque_rpm.append(None)

                elif (cond2 == []) and (cond3 == []) and (cond1 != []) and (cond_len == 1):
                    torque.append(float(nums[0]) * 9.8)
                    max_torque_rpm.append(None)
                elif (cond1 == []) and (cond3 == []) and (cond1 != []) and (cond_len == 2):
                    torque.append(float(nums[1]) * 9.8)
                    max_torque_rpm.append(None)

                elif (cond1 == []) and (cond3 == []) and (cond1 == []) and (cond_len == 2):
                    torque.append(nums[0])
                    max_torque_rpm.append(nums[1])
                elif (cond1 == []) and (cond3 == []) and (cond1 == []) and (cond_len == 3):
                    torque.append(nums[0])
                    max_torque_rpm.append(nums[2])
            elif cond0 == ['+']:
                nums = re.findall(r'\d+[.,\']\d+|\d+', df_train['torque'][i])
                cond1 = re.findall(r'kgm|KGM|Kgm', df_train['torque'][i])
                cond2 = re.findall(r'Nm|nm|NM', df_train['torque'][i])
                cond3 = re.findall(r'rpm|RPM|Rpm', df_train['torque'][i])
                cond_len = len(nums)
                if (cond3 != []) and (cond2 != []) and (cond_len == 3):
                    cond4 = re.findall(r',', df_train['torque'][i])
                    if cond4 != []:
                        torque.append(nums[0])
                        nums2 = nums[1].split(',')
                        nums3 = float(nums2[0] + nums2[1])
                        max_torque_rpm.append(nums3 + float(nums[2]))
                    elif cond4 == []:
                        torque.append(nums[0])
                        max_torque_rpm.append(float(nums[1]) + float(nums[2]))
        elif (df_train['torque'][i] == None):
            torque.append(None)
            max_torque_rpm.append(None)

    index4 = []
    for i in range(len(max_torque_rpm)):
        if (str(max_torque_rpm[i]) != 'None') and ((type(max_torque_rpm[i]) == str)):
            cond5 = re.findall(r',', max_torque_rpm[i])
            if cond5 != []:
                nums2 = max_torque_rpm[i].split(',')
                max_torque_rpm[i] = float(nums2[0] + nums2[1])
        index4.append(i)

    df_train['torque'] = torque
    df_train['max_torque_rpm'] = max_torque_rpm

    df_train['torque'] = df_train['torque'].astype(float)
    df_train['max_torque_rpm'] = df_train['max_torque_rpm'].astype(float)

    # --------------------------------------------------------------------------
    # Обработка пропусков
    medians = pd.read_csv('medians.csv')
    for col in medians.columns:
        for i in range(len(df_train)):
            if (str(df_train[col][i]) == 'nan') or (df_train[col][i] == None):
                df_train.loc[i, col] = medians.loc[0, col]

    # --------------------------------------------------------------------------
    df_train['engine'] = df_train['engine'].astype(int)
    df_train['seats'] = df_train['seats'].astype(int)
    df_train['torque'] = df_train['torque'].astype(float)
    df_train['max_torque_rpm'] = df_train['max_torque_rpm'].astype(float)


    # feature engineering
    df_train['year_sq'] = df_train['year'] ** 2
    age_of_car = (datetime.datetime.now().year - df_train['year']) + 1
    df_train['km_driven_per_year'] = df_train['km_driven'] / age_of_car
    df_train['specific power'] = df_train['max_power'] / df_train['engine']
    company = []
    for i in range(len(df_train)):
        comp = df_train['name'][i].split()[0]
        company.append(comp)

    df_train['company'] = company
    list_comp = ['Mitsubishi', 'Volvo', 'Jaguar', 'Force', 'Isuzu', 'Land', 'MG', 'Daewoo', 'Kia', 'Ambassador',
                     'Lexus', 'Peugeot']
    for i in range(len(df_train)):
        if df_train['company'][i] in list_comp:
            df_train.loc[i, 'company'] = 'other'
    owner_2 = []
    for i in range(len(df_train)):
        if (df_train['owner'][i] == 'First Owner') or (df_train['owner'][i] == 'Test Drive Car'):
            owner_2.append('First Owner & Test Drive Car')
        else:
            owner_2.append('Second & Above Owner')

    df_train['owner_2'] = owner_2


    # --------------------------------------------------------------------------
    # подготовка данных к построению модели
    X_cat = df_train[['fuel', 'seller_type', 'transmission', 'owner', 'company', 'owner_2', 'seats']]
    X_real = df_train[['year', 'km_driven', 'mileage', 'engine', 'max_power', 'torque',
                           'max_torque_rpm', 'year_sq', 'km_driven_per_year', 'specific power']]
    y_train = df_train['selling_price']

    #one_hot_enc = OneHotEncoder(drop='first', handle_unknown='ignore')
    #one_hot_enc.fit(X_cat)
    pkl_filename = "pickle_encoder.pkl"
    with open(pkl_filename, 'rb') as file:
        one_hot_enc = pickle.load(file)
    names_of_cat = []
    for i in range(len(one_hot_enc.categories_)):
        for j in range(1, len(one_hot_enc.categories_[i])):
            names_of_cat.append(one_hot_enc.categories_[i][j])

    X_dum01 = one_hot_enc.transform(X_cat)
    X_dum_train = pd.DataFrame(X_dum01.toarray(), columns=names_of_cat)
    X_dum = pd.concat([X_real, X_dum_train], axis=1)
    # X_num = X_dum.columns[:10]
    # X_cat = X_dum.columns[10:]
    # X_dum_v01 = X_dum[X_num]
    # X_cat_v01 = X_dum[X_cat]
    #
    # # стандартизируем вещественные признаки
    # #scaler = StandardScaler()
    # #scaler.fit(X_dum_v01)
    # pkl_filename = "pickle_scaler.pkl"
    # with open(pkl_filename, 'rb') as file:
    #     scaler = pickle.load(file)
    # X_real_stand = scaler.transform(X_dum_v01)
    # X_num_real_stand = pd.DataFrame(data=X_real_stand, columns=X_dum_v01.columns)
    #
    # X_dum_01 = pd.concat([X_num_real_stand, X_cat_v01], axis=1)

    # --------------------------------------------------------------------------
    # построение модели
    # model_ridge_1 = Ridge(alpha=1)
    # model_ridge = model_ridge_1.fit(X_dum_01, y_train)
    pkl_filename = "pickle_model_v02.pkl"
    with open(pkl_filename, 'rb') as file:
        model_ridge = pickle.load(file)

    y_pred_ridge = model_ridge.predict(X_dum)



    pd.set_option('display.max_columns', None)
    a = str(list(y_pred_ridge))

    return list(y_pred_ridge)