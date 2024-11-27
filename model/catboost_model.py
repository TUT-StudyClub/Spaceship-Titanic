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

#GrideSearchCat
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'iterations': [100, 200, 500],
    'depth': [4, 6, 8],
    'l2_leaf_reg': [1, 3, 5],
    'bagging_temperature': [0, 0.5, 1]
}

# GridSearchCVを実行
Cat_V_grid_search = GridSearchCV(
    estimator=CatBoostClassifier(verbose=0, random_state=42),
    param_grid=param_grid,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1,
    verbose=1
)

Cat_V_grid_search.fit(X_train_V, y_train_V)

Cat_V_grid_pred = Cat_V_grid_search.predict_proba(X_valid_V)
print(Cat_V_grid_pred[:5])

#trainに対して精度確認
score = roc_auc_score(y_valid_V,Cat_V_grid_pred[:, 1])
print(score)

Cat_V_grid_predict = Cat_V_grid_search.predict_proba(test_V)
print(Cat_V_grid_predict[:5])

Cat_V_grid_predict = Cat_V_grid_search.predict_proba(test_V)[:,1]
Cat_V_grid_predict_test = (Cat_V_grid_predict >= 0.5)
print(Cat_V_grid_predict_test[:5])

submit['Transported'] = Cat_V_grid_predict_test
submit.head()
#ファイル化
submit.to_csv("submission.csv", index=False)