import pandas as pd
import numpy as np
import seaborn as sns
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# データの読み込み
train = pd.read_csv("data/train.csv")
train_V = pd.read_csv("data/train_v2.csv")
test_V = pd.read_csv("data/test_v2.csv")
submit = pd.read_csv("data/sample_submission.csv")

train_V.head()

# ターゲット抽出
target = train['Transported'].astype(int)  # ターゲット（0と1に変換）

#Yさんデータの分割
X_train_V, X_valid_V, y_train_V, y_valid_V = train_test_split(train_V, target, random_state = 42)
print(X_train_V.shape, X_valid_V.shape, y_train_V.shape, y_valid_V.shape)

param_grid = {
    'num_leaves': [15, 31, 63],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 500],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.7, 1.0]
}

Light_V_grid_search = GridSearchCV(
    estimator=LGBMClassifier(random_state=42),
    param_grid=param_grid,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1,
    verbose=1
)
Light_V_grid_search.fit(X_train_V, y_train_V)

Light_V_grid_pred = Light_V_grid_search.predict_proba(X_valid_V)
print(Light_V_grid_pred[:5])

#trainに対して精度確認
score = roc_auc_score(y_valid_V,Light_V_grid_pred[:, 1])
print(score)

Light_V_grid_predict = Light_V_grid_search.predict_proba(test_V)
print(Light_V_grid_predict[:5])

Light_V_grid_predict = Light_V_grid_search.predict_proba(test_V)[:,1]
Light_V_grid_predict_test = (Light_V_grid_predict >= 0.5)
print(Light_V_grid_predict_test[:5])

submit['Transported'] = Light_V_grid_predict_test
submit.head()
#ファイル化
submit.to_csv("lG.csv", index=False)