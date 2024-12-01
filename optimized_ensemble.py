import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score

# データの読み込み
train = pd.read_csv("data/train.csv")
train_V = pd.read_csv("data/train_v2.csv")
test_V = pd.read_csv("data/test_v2.csv")
submit = pd.read_csv("data/sample_submission.csv")

# ターゲット抽出
target = train['Transported'].astype(int)

# データの分割
X_train_V, X_valid_V, y_train_V, y_valid_V = train_test_split(train_V, target, random_state=42)
print(X_train_V.shape, X_valid_V.shape, y_train_V.shape, y_valid_V.shape)

# 前処理
numeric_features = X_train_V.select_dtypes(include=['float64', 'int64']).columns
categorical_features = X_train_V.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', 'passthrough')  # 必要に応じてOneHotEncoderに変更
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Base models
rf = RandomForestClassifier(random_state=42)
lgbm = LGBMClassifier(random_state=42)
cat = CatBoostClassifier(verbose=0, random_state=42)
# xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# ハイパーパラメータチューニング
rf_params = {'n_estimators': [100, 200], 'max_depth': [None, 10]}
lgbm_params = {'learning_rate': [0.1, 0.01], 'n_estimators': [100, 200]}
cat_params = {'learning_rate': [0.1, 0.01], 'depth': [6, 8]}
# xgb_params = {'learning_rate': [0.1, 0.01], 'max_depth': [6, 8], 'n_estimators': [100, 200]}

rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='roc_auc', n_jobs=-1).fit(X_train_V, y_train_V)
lgbm_grid = GridSearchCV(lgbm, lgbm_params, cv=3, scoring='roc_auc', n_jobs=-1).fit(X_train_V, y_train_V)
cat_grid = GridSearchCV(cat, cat_params, cv=3, scoring='roc_auc', n_jobs=-1).fit(X_train_V, y_train_V)
# xgb_grid = GridSearchCV(xgb, xgb_params, cv=3, scoring='roc_auc', n_jobs=-1).fit(X_train_V, y_train_V)

# 最適化モデル
best_rf = rf_grid.best_estimator_
best_lgbm = lgbm_grid.best_estimator_
best_cat = cat_grid.best_estimator_
# best_xgb = xgb_grid.best_estimator_

# Meta-model
stacking_model = StackingClassifier(
    estimators=[
        ('rf', best_rf),
        ('lgbm', best_lgbm),
        ('cat', best_cat),
        # ('xgb', best_xgb)
    ],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,
    n_jobs=-1
)

# Stackingモデル学習
stacking_model.fit(X_train_V, y_train_V)

# 検証データでの予測
stacking_V_pred = stacking_model.predict_proba(X_valid_V)[:, 1]
score = roc_auc_score(y_valid_V, stacking_V_pred)
print(f"Validation ROC AUC Score: {score:.4f}")

# テストデータでの予測
stacking_V_grid_predict = stacking_model.predict_proba(test_V)[:, 1]
stacking_V_grid_predict_test = (stacking_V_grid_predict >= 0.5).astype(bool)

# 提出ファイル作成
submit['Transported'] = stacking_V_grid_predict_test
submit.to_csv("optimized_ensemble.csv", index=False)
print("Submission file 'optimized_ensemble.csv' created.")