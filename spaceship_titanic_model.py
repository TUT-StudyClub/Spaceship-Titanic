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
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# データの読み込み
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
train_G = pd.read_csv("data/train_G1.csv")
test_G = pd.read_csv("data/test_G1.csv")
train_V = pd.read_csv("data/train_v2.csv")
test_V = pd.read_csv("data/test_v2.csv")
submit = pd.read_csv("data/sample_submission.csv")

# ターゲット抽出
target = train['Transported'].astype(int)  # ターゲット（0と1に変換）

# 特徴量削除
train = train.drop(columns=['Transported','PassengerId','Name','Cabin'],axis=1)
train_G = train_G.drop(columns=['Transported','PassengerId','Name'],axis=1)

test = test.drop(columns=['PassengerId','Name','Cabin'],axis=1)
test_G = test_G.drop(columns=['PassengerId','Name'],axis=1)

#必要に応じてカテゴリデータのエンコーディング
train = pd.get_dummies(train, drop_first=True)
test = pd.get_dummies(test, drop_first=True)

#欠損値中央値埋め
imputer = SimpleImputer(strategy="median")
train = pd.DataFrame(imputer.fit_transform(train), columns=train.columns)
test = pd.DataFrame(imputer.fit_transform(test), columns=test.columns)
train_G = pd.DataFrame(imputer.fit_transform(train_G), columns=train_G.columns)
test_G = pd.DataFrame(imputer.fit_transform(test_G), columns=test_G.columns)

#前処理無し
X_train, X_valid, y_train, y_valid = train_test_split(train, target, random_state = 42)

#Rさん
X_train_G, X_valid_G, y_train_G, y_valid_G = train_test_split(train_G, target, random_state = 42)

#Yさん
X_train_V, X_valid_V, y_train_V, y_valid_V = train_test_split(train_V, target, random_state = 42)

#勾配ブースティング系
# CatBoost
Cat = CatBoostClassifier(verbose=0, random_state=42)
Cat.fit(X_train, y_train)

Cat_G = CatBoostClassifier(verbose=0, random_state=42)
Cat_G.fit(X_train_G, y_train_G)

Cat_V = CatBoostClassifier(verbose=0, random_state=42)
Cat_V.fit(X_train_V, y_train_V)

# 検証データで予測
Cat_pred = Cat.predict_proba(X_valid)[:, 1]  # 検証データの陽性クラスの確率
Cat_pred_G = Cat_G.predict_proba(X_valid_G)[:, 1]
Cat_pred_V = Cat_V.predict_proba(X_valid_V)[:, 1]

# スコアの計算
Cat_roc_auc = roc_auc_score(y_valid, Cat_pred)
Cat_roc_auc_G = roc_auc_score(y_valid_G, Cat_pred_G)
Cat_roc_auc_V = roc_auc_score(y_valid_V, Cat_pred_V)

#上昇率
increase_Cat_G = ((Cat_roc_auc_G - Cat_roc_auc) / Cat_roc_auc) * 100
increase_Cat_V = ((Cat_roc_auc_V - Cat_roc_auc) / Cat_roc_auc) * 100

# LightGBM
Light = LGBMClassifier(random_state=42)
Light.fit(X_train, y_train)

Light_G = LGBMClassifier(random_state=42)
Light_G.fit(X_train_G, y_train_G)

Light_V = LGBMClassifier(random_state=42)
Light_V.fit(X_train_V, y_train_V)

# 検証データで予測
Light_pred = Light.predict_proba(X_valid)[:, 1]  # 検証データの陽性クラスの確率
Light_pred_G = Light_G.predict_proba(X_valid_G)[:, 1]
Light_pred_V = Light_V.predict_proba(X_valid_V)[:, 1]

# スコアの計算
Light_roc_auc = roc_auc_score(y_valid, Light_pred)
Light_roc_auc_G = roc_auc_score(y_valid_G, Light_pred_G)
Light_roc_auc_V = roc_auc_score(y_valid_V, Light_pred_V)

#上昇率
increase_Light_G = ((Light_roc_auc_G - Light_roc_auc) / Light_roc_auc) * 100
increase_Light_V = ((Light_roc_auc_V - Light_roc_auc) / Light_roc_auc) * 100

# XGBoost
XG = XGBClassifier(eval_metric='logloss', random_state=42)
XG.fit(X_train, y_train)

XG_G = XGBClassifier(eval_metric='logloss', random_state=42)
XG_G.fit(X_train_G, y_train_G)

XG_V = XGBClassifier(eval_metric='logloss', random_state=42)
XG_V.fit(X_train_V, y_train_V)

# 検証データで予測
XG_pred = XG.predict_proba(X_valid)[:, 1]  # 検証データの陽性クラスの確率
XG_pred_G = XG_G.predict_proba(X_valid_G)[:, 1]
XG_pred_V = XG_V.predict_proba(X_valid_V)[:, 1]

# スコアの計算
XG_roc_auc = roc_auc_score(y_valid, XG_pred)
XG_roc_auc_G = roc_auc_score(y_valid_G, XG_pred_G)
XG_roc_auc_V = roc_auc_score(y_valid_V, XG_pred_V)

#上昇率
increase_XG_G = ((XG_roc_auc_G - XG_roc_auc) / XG_roc_auc) * 100
increase_XG_V = ((XG_roc_auc_V - XG_roc_auc) / XG_roc_auc) * 100

# ランダムフォレスト
RF = RandomForestClassifier(random_state=42)
RF.fit(X_train, y_train)

RF_G = RandomForestClassifier(random_state=42)
RF_G.fit(X_train_G, y_train_G)

RF_V = RandomForestClassifier(random_state=42)
RF_V.fit(X_train_V, y_train_V)

# 検証データで予測
RF_pred = RF.predict_proba(X_valid)[:, 1]  # 検証データの陽性クラスの確率
RF_pred_G = RF_G.predict_proba(X_valid_G)[:, 1]
RF_pred_V = RF_V.predict_proba(X_valid_V)[:, 1]

# スコアの計算
RF_roc_auc = roc_auc_score(y_valid, RF_pred)
RF_roc_auc_G = roc_auc_score(y_valid_G, RF_pred_G)
RF_roc_auc_V = roc_auc_score(y_valid_V, RF_pred_V)

#上昇率
increase_RF_G = ((RF_roc_auc_G - RF_roc_auc) / RF_roc_auc) * 100
increase_RF_V = ((RF_roc_auc_V - RF_roc_auc) / RF_roc_auc) * 100

# ロジスティック回帰
LR = LogisticRegression(max_iter=1000, random_state=42)
LR.fit(X_train, y_train)

LR_G = LogisticRegression(max_iter=1000, random_state=42)
LR_G.fit(X_train_G, y_train_G)

LR_V = LogisticRegression(max_iter=1000, random_state=42)
LR_V.fit(X_train_V, y_train_V)

# 検証データで予測
LR_pred = LR.predict_proba(X_valid)[:, 1]  # 検証データの陽性クラスの確率
LR_pred_G = LR_G.predict_proba(X_valid_G)[:, 1]
LR_pred_V = LR_V.predict_proba(X_valid_V)[:, 1]

# スコアの計算
LR_roc_auc = roc_auc_score(y_valid, LR_pred)
LR_roc_auc_G = roc_auc_score(y_valid_G, LR_pred_G)
LR_roc_auc_V = roc_auc_score(y_valid_V, LR_pred_V)

#上昇率
increase_LR_G = ((LR_roc_auc_G - LR_roc_auc) / LR_roc_auc) * 100
increase_LR_V = ((LR_roc_auc_V - LR_roc_auc) / LR_roc_auc) * 100

# 決定木
DT = DecisionTreeClassifier(
    max_depth=5,           # 木の最大深さ
    min_samples_split=10,  # 分割される最小サンプル数
    random_state=42        # 再現性のための乱数シード
)
DT.fit(X_train, y_train)

DT_G = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    random_state=42
)
DT_G.fit(X_train_G, y_train_G)

DT_V = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    random_state=42
)
DT_V.fit(X_train_V, y_train_V)

# 検証データで予測
DT_pred = DT.predict_proba(X_valid)[:, 1]  # 検証データの陽性クラスの確率
DT_pred_G = DT_G.predict_proba(X_valid_G)[:, 1]
DT_pred_V = DT_V.predict_proba(X_valid_V)[:, 1]

# スコアの計算
DT_roc_auc = roc_auc_score(y_valid, DT_pred)
DT_roc_auc_G = roc_auc_score(y_valid_G, DT_pred_G)
DT_roc_auc_V = roc_auc_score(y_valid_V, DT_pred_V)

#上昇率
increase_DT_G = ((DT_roc_auc_G - DT_roc_auc) / DT_roc_auc) * 100
increase_DT_V = ((DT_roc_auc_V - DT_roc_auc) / DT_roc_auc) * 100

# サポートベクターマシン
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)

SVM = SVC(kernel='rbf', probability=True, C=1.0, gamma='scale', random_state=42)
SVM.fit(X_train, y_train)

SVM_G = SVC(kernel='rbf', probability=True, C=1.0, gamma='scale', random_state=42)
SVM_G.fit(X_train_G, y_train_G)

SVM_V = SVC(kernel='rbf', probability=True, C=1.0, gamma='scale', random_state=42)
SVM_V.fit(X_train_V, y_train_V)

# 検証データで予測
SVM_pred = SVM.predict_proba(X_valid)[:, 1]  # 検証データの陽性クラスの確率
SVM_pred_G = SVM_G.predict_proba(X_valid_G)[:, 1]
SVM_pred_V = SVM_V.predict_proba(X_valid_V)[:, 1]

# スコアの計算
SVM_roc_auc = roc_auc_score(y_valid, SVM_pred)
SVM_roc_auc_G = roc_auc_score(y_valid_G, SVM_pred_G)
SVM_roc_auc_V = roc_auc_score(y_valid_V, SVM_pred_V)

#上昇率
increase_SVM_G = ((SVM_roc_auc_G - SVM_roc_auc) / SVM_roc_auc) * 100
increase_SVM_V = ((SVM_roc_auc_V - SVM_roc_auc) / SVM_roc_auc) * 100

print("前処理1回目")
print("CatBoost")
print(f"前処理無し:{Cat_roc_auc}")
print(f'Rさん:{Cat_roc_auc_G}, 向上率:{round(increase_Cat_G,1)}%')
print(f'Yさん:{Cat_roc_auc_V}, 向上率:{round(increase_Cat_V,1)}%\n')

print("LightGBM")
print(f"前処理無し:{Light_roc_auc}")
print(f'Rさん:{Light_roc_auc_G}, 向上率:{round(increase_Light_G,1)}%')
print(f'Yさん:{Light_roc_auc_V}, 向上率:{round(increase_Light_V,1)}%\n')

print("XGBoost")
print(f"前処理無し:{XG_roc_auc}")
print(f'Rさん:{XG_roc_auc_G}, 向上率:{round(increase_XG_G,1)}%')
print(f'Yさん:{XG_roc_auc_V}, 向上率:{round(increase_XG_V,1)}%\n')

print("ランダムフォレスト")
print(f"前処理無し:{RF_roc_auc}")
print(f'Rさん:{RF_roc_auc_G}, 向上率:{round(increase_RF_G,1)}%')
print(f'Yさん:{RF_roc_auc_V}, 向上率:{round(increase_RF_V,1)}%\n')

print("ロジスティック回帰")
print(f"前処理無し:{LR_roc_auc}")
print(f'Rさん:{LR_roc_auc_G}, 向上率:{round(increase_LR_G,1)}%')
print(f'Yさん:{LR_roc_auc_V}, 向上率:{round(increase_LR_V,1)}%\n')

print("決定木")
print(f"前処理無し:{DT_roc_auc}")
print(f'Rさん:{DT_roc_auc_G}, 向上率:{round(increase_DT_G,1)}%')
print(f'Yさん:{DT_roc_auc_V}, 向上率:{round(increase_DT_V,1)}%\n')

print("サポートベクターマシン")
print(f"前処理無し:{SVM_roc_auc}")
print(f'Rさん:{SVM_roc_auc_G}, 向上率:{round(increase_SVM_G,1)}%')
print(f'Yさん:{SVM_roc_auc_V}, 向上率:{round(increase_SVM_V,1)}%\n')

#テストデータに対して予測
predict_Cat = Cat.predict_proba(test)
Cat_pred_test = (predict_Cat >= 0.5) #0.5以上をTrue
predict_Cat_G = Cat_G.predict_proba(test_G)
Cat_G_pred_test = (predict_Cat_G >= 0.5)
predict_Cat_V = Cat_V.predict_proba(test_V)
Cat_V_pred_test = (predict_Cat_V >= 0.5)

predict_Light = Light.predict_proba(test)
Light_pred_test = (predict_Light >= 0.5)
predict_Light_G = Light_G.predict_proba(test_G)
Light_G_pred_test = (predict_Light_G >= 0.5)
predict_Light_V = Light_V.predict_proba(test_V)
Light_V_pred_test = (predict_Light_V >= 0.5)

predict_XG = XG.predict_proba(test)
XG_pred_test = (predict_XG >= 0.5)
predict_XG_G = XG_G.predict_proba(test_G)
XG_G_pred_test = (predict_XG_G >= 0.5)
predict_XG_V = XG_V.predict_proba(test_V)
XG_V_pred_test = (predict_XG_V >= 0.5)

predict_RF = RF.predict_proba(test)
RF_pred_test = (predict_RF >= 0.5)
predict_RF_G = RF_G.predict_proba(test_G)
RF_G_pred_test = (predict_RF_G >= 0.5)
predict_RF_V = RF_V.predict_proba(test_V)
RF_V_pred_test = (predict_RF_V >= 0.5)

predict_DT = DT.predict_proba(test)
DT_pred_test = (predict_DT >= 0.5)
predict_DT_G = DT_G.predict_proba(test_G)
DT_G_pred_test = (predict_DT_G >= 0.5)
predict_DT_V = DT_V.predict_proba(test_V)
DT_V_pred_test = (predict_DT_V >= 0.5)

submit['Transported'] = Cat_pred_test[:, 1]
submit.head()

#submit.to_csv("submission.csv", index=False)
