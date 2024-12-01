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
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
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

# Base models
estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('lgbm', LGBMClassifier(random_state=42)),
    ('cat', CatBoostClassifier(verbose=0, random_state=42))
]

# Meta-model
stacking_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5,
    n_jobs=-1
)

# 訓練データに適用
stacking_model.fit(X_train_V, y_train_V)

stacking_V_pred = stacking_model.predict_proba(X_valid_V)
#print(stacking_V_pred[:5])

#精度確認
score = roc_auc_score(y_valid_V,stacking_V_pred[:, 1])
print(score)

stacking_V_grid_predict = stacking_model.predict_proba(test_V)
#print(stacking_V_grid_predict[:5])

stacking_V_grid_predict = stacking_model.predict_proba(test_V)[:,1]
stacking_V_grid_predict_test = (stacking_V_grid_predict >= 0.5)
#print(stacking_V_grid_predict_test[:10])

submit['Transported'] = stacking_V_grid_predict_test
submit.head()

submit.to_csv("ensemble.csv", index=False)