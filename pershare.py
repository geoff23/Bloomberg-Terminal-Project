import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV

from sklearn.model_selection import cross_val_score 

og = pd.read_csv('psdata.csv', header=1)
df = og.drop('ID', axis=1)
df = df.drop('cur_mkt_cap()', axis=1)
df = df.replace('#N/A', np.NaN)

names = list(df.columns)
names.remove('px_last()')
def plot_coefs(linear_model):
    plt.plot(range(len(names)), linear_model.coef_)
    plt.xticks(range(len(names)), names, rotation=90)
    plt.ylabel('Coefficients')
    plt.tight_layout()
    plt.show()

def linear_loocv():
    new_df = df.dropna()
    table = []
    for index, row in new_df.iterrows():
        ticker = og.loc[index]['ID']
        market_cap = og.loc[index]['cur_mkt_cap()']
        print(ticker)
        train_df = new_df.drop(index)
        test_df = new_df.loc[index:index+1]
        X_train = train_df.drop('px_last()', axis=1)
        y_train = train_df['px_last()'].values
        X_test = test_df.drop('px_last()', axis=1)
        y_test = test_df['px_last()'].values
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        prediction = reg.predict(X_test)
        table.append([ticker, market_cap, y_test[0], prediction[0], prediction[0]/y_test[0]])
    summary = pd.DataFrame(table, columns = ['Ticker', 'Market Cap', 'Actual', 'Predicted', 'Ratio'])
    summary.to_csv('pspredictions.csv', index = False)

def linear_pipeline_loocv():
    table = []
    for index, row in df.iterrows():
        train_df = df.drop(index)
        test_df = df.loc[index:index+1]
        X_train = train_df.drop('cur_mkt_cap()', axis=1)
        y_train = train_df['cur_mkt_cap()'].values
        X_test = test_df.drop('cur_mkt_cap()', axis=1)
        y_test = test_df['cur_mkt_cap()'].values
        steps = [('imputation', SimpleImputer(missing_values=np.nan, strategy='median')),
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())]
        pipeline = Pipeline(steps)
        pipeline.fit(X_train, y_train)
        prediction = pipeline.predict(X_test)
        table.append([og.loc[index]['ID'], y_test[0], prediction[0], prediction[0]/y_test[0]])
    summary = pd.DataFrame(table, columns = ['Ticker', 'Actual', 'Predicted', 'Ratio'])
    summary.to_csv('predictions.csv', index = False)

linear_loocv()