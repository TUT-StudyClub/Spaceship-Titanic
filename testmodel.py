import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import lightgbm as lgb

# CSVファイルのパス
file_path_train = "C:/Users/c0b2217150/Documents/ml/spaceship/train_v2.csv"
file_path_test = "C:/Users/c0b2217150/Documents/ml/spaceship/test_v2.csv"
file_path_sub = "C:/Users/c0b2217150/Documents/ml/spaceship/sample_submission.csv"

# CSVファイルを読み込む
train = pd.read_csv(file_path_train)
test = pd.read_csv(file_path_test)
submit = pd.read_csv(file_path_sub)

# 特徴量とターゲットの分割
X = train.drop(columns=['Transported'])  # 特徴量
y = train['Transported'].astype(int)  # ターゲット（0と1に変換）

# データを訓練データと検証データに分割
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# LightGBM のパラメータチューニング
param_grid = {
    'num_leaves': [10, 35, 50],
    'max_depth': [-1, 100],
    'learning_rate': [0.1, 0.2],
    'n_estimators': [100, 200],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [0, 0.1, 1]
}

# GridSearchCVによるパラメータチューニング
grid_search = GridSearchCV(
    estimator=lgb.LGBMClassifier(random_state=42),
    param_grid=param_grid,
    scoring='accuracy',  # 評価指標 (必要に応じて変更可能)
    cv=5,                # クロスバリデーションの分割数
    verbose=1,
    n_jobs=-1            # 並列処理
)

# モデルの学習
grid_search.fit(X_train, y_train)

# 最適なパラメータの取得
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation accuracy: ", grid_search.best_score_)

# 最適モデルでのテスト
best_model = grid_search.best_estimator_
pred_proba = best_model.predict_proba(X_val)  # 確率予測を取得
pred = best_model.predict(X_val)  # クラス予測を取得

# AUCスコアの計算
score_auc = roc_auc_score(y_val, pred_proba[:, 1])  # 正しい確率列にアクセス
print(f"AUCスコア: {score_auc}")

# F1スコアの計算
y_pred = (pred_proba[:, 1] >= 0.5).astype(int)  # 確率を基にしたクラス分け
score_f1 = f1_score(y_val, y_pred)
print(f"F1スコア: {score_f1}")

# 精度（Accuracy）の計算
accuracy = accuracy_score(y_val, y_pred)
print(f"精度（Accuracy）: {accuracy}")

# テストデータを用いて予測します。
predict = best_model.predict_proba(test)

# 提出用のデータフレームに予測を追加
submit['Transported'] = predict[:, 1] >= 0.5

# CSVファイルとして出力
output_path = "C:/Users/c0b2217150/Documents/ml/spaceship/first_2.csv"
submit.to_csv(output_path, index=False)  # index=Falseで行番号を含めない
print(f"CSVファイルを出力しました: {output_path}")
