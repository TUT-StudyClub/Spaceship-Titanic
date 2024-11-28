import pandas as pd
import numpy as np

df_train = pd.read_csv('/Users/yamadayuuhei/Github/space_titanic/train.csv')
df_test = pd.read_csv('/Users/yamadayuuhei/Github/space_titanic/test.csv')

""""
trainデータの前処理
"""
train = df_train.copy()

## HomePlanetの数値化と欠損値補完
title_mapping = {"Europa": 1, "Earth": 2, "Mars": 3}
train['HomePlanet'] = train['HomePlanet'].map(title_mapping)
train['HomePlanet'] = train['HomePlanet'].fillna(0)

## Cabinの分割と各部分の数値化
train[['Cabin_1', 'Cabin_2', 'Cabin_3']] = train['Cabin'].str.split('/', expand=True)
train.drop('Cabin', axis=1, inplace=True)
cabin_1_mapping = {"B": 1, "F": 2, "A": 3, "G": 4, "E": 5, "D": 6, "C": 7, "T": 8} # 文字列を数値にマッピング  
train['Cabin_1'] = train['Cabin_1'].map(cabin_1_mapping)
train['Cabin_1'] = train['Cabin_1'].fillna(0)
cabin_3_mapping = {"P": 1, "S": 2}
train['Cabin_3'] = train['Cabin_3'].map(cabin_3_mapping)
train['Cabin_3'] = train['Cabin_3'].fillna(0)
train['Cabin_2'] = train['Cabin_2'].fillna(9999)# 異常値を一時的に使用することで、後で簡単に検出して処理
train['Cabin_2'] = train['Cabin_2'].astype(int) + 1
train['Cabin_2'] = train['Cabin_2'].replace(10000, 0)
## 新しい特徴量Fare作成
train['Fare'] = train[['FoodCourt', 'RoomService', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)
cols = ['FoodCourt', 'RoomService', 'ShoppingMall', 'Spa', 'VRDeck']
for col in cols:
    train[col] = train[col].fillna(train[col].median())

## 欠損値補完
train.loc[train["Transported"] == 1, "Age"] = train[train["Transported"] == 1]["Age"].fillna(train[train["Transported"] == 1]["Age"].mean())
train.loc[train["Transported"] == 0, "Age"] = train[train["Transported"] == 0]["Age"].fillna(train[train["Transported"] == 0]["Age"].mean())
train.drop('Destination', axis=1, inplace=True)

vip_mapping = {False: 1, True: 2}

train['VIP'] = train['VIP'].map(vip_mapping)
train['VIP'] = train['VIP'].fillna(0)

cryo_mapping = {False: 1, True: 2}
train['CryoSleep'] = train['CryoSleep'].map(cryo_mapping)
train['CryoSleep'] = train['CryoSleep'].fillna(0)

train.drop(['Name', 'Transported', 'PassengerId'], axis = 1, inplace = True)

""""
testデータの前処理
"""
test = df_test.copy()

title_mapping = {"Europa": 1, "Earth": 2, "Mars": 3}
test['HomePlanet'] = test['HomePlanet'].map(title_mapping)
test['HomePlanet'] = test['HomePlanet'].fillna(0)

test[['Cabin_1', 'Cabin_2', 'Cabin_3']] = test['Cabin'].str.split('/', expand=True)
test.drop('Cabin', axis=1, inplace=True)
cabin_1_mapping = {"B": 1, "F": 2, "A": 3, "G": 4, "E": 5, "D": 6, "C": 7, "T": 8}
test['Cabin_1'] = test['Cabin_1'].map(cabin_1_mapping)
test['Cabin_1'] = test['Cabin_1'].fillna(0)
cabin_3_mapping = {"P": 1, "S": 2}
test['Cabin_3'] = test['Cabin_3'].map(cabin_3_mapping)
test['Cabin_3'] = test['Cabin_3'].fillna(0)
test['Cabin_2'] = test['Cabin_2'].fillna(9999)
test['Cabin_2'] = test['Cabin_2'].astype(int) + 1
test['Cabin_2'] = test['Cabin_2'].replace(10000, 0)

test['Fare'] = test[['FoodCourt', 'RoomService', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)
cols = ['FoodCourt', 'RoomService', 'ShoppingMall', 'Spa', 'VRDeck']
for col in cols:
    test[col] = test[col].fillna(test[col].median())

test['Age'] = test['Age'].fillna(train["Age"].mean())
test.drop('Destination', axis=1, inplace=True)

vip_mapping = {False: 1, True: 2}
test['VIP'] = test['VIP'].map(vip_mapping)
test['VIP'] = test['VIP'].fillna(0)

cryo_mapping = {False: 1, True: 2}
test['CryoSleep'] = test['CryoSleep'].map(cryo_mapping)
test['CryoSleep'] = test['CryoSleep'].fillna(0)

test.drop(['Name', 'PassengerId'], axis = 1, inplace = True)