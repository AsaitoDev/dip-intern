import streamlit as st
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


st.title("求人原稿に対する応募数を予測する")

st.write("【開発環境】MacBook Pro・VScode")
st.write("【テスト環境】Google Chrome")
st.write("【使用したフレームワーク】streamlit")
st.write("【注意】アプリ起動に少々お時間を頂戴いたします。忙しい中お時間取らせてしまい申し訳ございません。")

#回帰モデルの作成

#データの読み込み
train_y = pd.read_csv("train_y.csv", encoding="utf-8")
train_X = pd.read_csv("train_x.csv", encoding="utf-8")

#train_y["お仕事No."]をdropする
train_y = train_y.drop(["お仕事No."], axis=1)

#train_xのデータ整形（空白だけ・不要？なカラムの削除）
train_X = train_X.dropna(axis=1, how='all')
train_X = train_X.fillna(0)
train_X = train_X.drop(["掲載期間　開始日", "掲載期間　終了日", "期間・時間　勤務開始日","動画コメント", "（派遣）応募後の流れ", "拠点番号", "（派遣先）勤務先写真ファイル名", "動画タイトル", "会社概要　業界コード", "派遣会社のうれしい特典", "動画ファイル名"], axis=1)

#train_yとtrain_Xを連結して学習データとして利用する
train = pd.concat([train_y.reset_index(drop=True), train_X.reset_index(drop=True)], axis=1)

#データ変換と分割
y = train["応募数 合計"]
X = train.drop(["お仕事No.", "応募数 合計"], axis=1)

y_array = np.array(y)
X_array = np.array(X)

#文章のデータ整形が難しかったので、文章のカラムを一括に学習データから外します。
X_array = X.select_dtypes(exclude="object")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_array, y_array, test_size=0.4, random_state=0)

#回帰モデル作成
rfr = RandomForestRegressor(random_state=0)
rfr.fit(X_train, y_train)

y_pred = rfr.predict(X_test)
# print(y_pred.shape)

mse = np.sqrt(mean_squared_error(y_pred, y_test))
st.write(f"Mean Squared Error:{mse:.3}")

#ファイルがアップロードされるまでの処理を書く
uploaded_file = st.file_uploader("Choose a file", type=["csv"])

if uploaded_file is not None:
    test_X = pd.read_csv(uploaded_file).dropna(axis=1, how="all").fillna(0)

    #応募数合計の予測をするためのデータ成形
    test_X = test_X.drop(["掲載期間　開始日", "掲載期間　終了日", "期間・時間　勤務開始日","動画コメント", "（派遣）応募後の流れ", "拠点番号", "（派遣先）勤務先写真ファイル名", "動画タイトル", "会社概要　業界コード", "派遣会社のうれしい特典", "動画ファイル名"], axis=1)

    #文章のデータ整形が難しかったので、文章のカラムを一括に学習データから外します。
    test_X2 = test_X.drop(["お仕事No."], axis=1)
    test_X2 = test_X2.select_dtypes(exclude="object")

    #応募数合計の予測・データの予測
    test_pred = rfr.predict(test_X2)
    print(test_pred.shape)

    test_pred = pd.DataFrame(test_pred, columns=["応募数 合計"])
    result = pd.concat([test_X["お仕事No."], test_pred], axis=1)
    result = result.reset_index(drop=True)

    st.dataframe(result)
    st.download_button(label = '予測結果をダウンロード', data=result.to_csv(index=False).encode('utf-8'), file_name="result.csv")