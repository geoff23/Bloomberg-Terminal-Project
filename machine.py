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

og = pd.read_csv('data.csv')
df = og.drop('ID', axis=1)
df = df.replace('#N/A', np.NaN)

names = list(df.columns)
names.remove('cur_mkt_cap()')
def plot_coefs(linear_model):
    plt.plot(range(len(names)), linear_model.coef_)
    plt.xticks(range(len(names)), names, rotation=90)
    plt.ylabel('Coefficients')
    plt.tight_layout()
    plt.show()

def predict(reg, df=df):
    X = df.drop('cur_mkt_cap()', axis=1)
    predictions = reg.predict(X)
    table = []
    for i in X.index.values:
        ticker = og.loc[i]['ID']
        actual = og.loc[i]['cur_mkt_cap()']
        predicted = predictions[list(X.index.values).index(i)]
        table.append([ticker, actual, predicted, predicted/actual])
    summary = pd.DataFrame(table, columns = ['Ticker', 'Actual', 'Predicted', 'Ratio'])
    summary.to_csv('predictions.csv', index = False)

def linear():
    new_df = df.dropna()
    X = new_df.drop('cur_mkt_cap()', axis=1)
    y = new_df['cur_mkt_cap()'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    reg = LinearRegression()
    reg.fit(X_train, y_train)
    print("R^2 (training):", reg.score(X_train, y_train))
    print("R^2: (testing)", reg.score(X_test, y_test))
    plot_coefs(reg)
    predict(reg, df=new_df)

def linear_pipeline():
    steps = [('imputation', SimpleImputer(missing_values=np.nan, strategy='median')),
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())]
    pipeline = Pipeline(steps)
    X = df.drop('cur_mkt_cap()', axis=1)
    y = df['cur_mkt_cap()'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    pipeline.fit(X_train, y_train)
    print("R^2 (training):", pipeline.score(X_train, y_train))
    print("R^2: (testing)", pipeline.score(X_test, y_test))
    plot_coefs(pipeline['regressor'])
    predict(pipeline)

def display_plot(scores, alpha_space):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(alpha_space, scores)

    ax.axhline(np.max(scores), linestyle='--', color='.5')
    ax.set_ylabel('Score')
    ax.set_xlabel('Alpha')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()

def lasso_ridge(reg_type, alpha_space):
    new_df = df.dropna()
    X = new_df.drop('cur_mkt_cap()', axis=1)
    y = new_df['cur_mkt_cap()'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    X_train_train, X_train_test, y_train_train, y_train_test = train_test_split(X_train, y_train, test_size = 0.25, random_state = 42)
    train_scores = np.empty(len(alpha_space))
    test_scores = np.empty(len(alpha_space))

    for i, a in enumerate(alpha_space):
        if reg_type == 'ridge':
            reg = Ridge(alpha=a)
        else:
            reg = Lasso(alpha=a)
        reg.fit(X_train_train, y_train_train)
        train_scores[i] = reg.score(X_train_train, y_train_train)
        test_scores[i] = reg.score(X_train_test, y_train_test)

    display_plot(train_scores, alpha_space)
    display_plot(test_scores, alpha_space)
    print("Best alpha:", alpha_space[np.argmax(test_scores)])

    if reg_type == 'ridge':
        reg = Ridge(alpha=alpha_space[np.argmax(test_scores)])
    else:
        reg = Lasso(alpha=alpha_space[np.argmax(test_scores)])
    reg.fit(X_train, y_train)
    print("R^2 (training):", reg.score(X_train, y_train))
    print("R^2: (testing)", reg.score(X_test, y_test))
    plot_coefs(reg)
    predict(reg, df = new_df)

def lasso_ridge_pipeline(reg_type, alpha_space):
    X = df.drop('cur_mkt_cap()', axis=1)
    y = df['cur_mkt_cap()'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    X_train_train, X_train_test, y_train_train, y_train_test = train_test_split(X_train, y_train, test_size = 0.25, random_state = 42)
    
    train_scores = np.empty(len(alpha_space))
    test_scores = np.empty(len(alpha_space))

    for i, a in enumerate(alpha_space):
        if reg_type == 'ridge':
            reg = Ridge(alpha = a)
        else:
            reg = Lasso(alpha = a)
        steps = [('imputation', SimpleImputer(missing_values=np.nan, strategy='median')),
                ('scaler', StandardScaler()),
                ('regressor', reg)]
        pipeline = Pipeline(steps)
        pipeline.fit(X_train_train, y_train_train)
        train_scores[i] = pipeline.score(X_train_train, y_train_train)
        test_scores[i] = pipeline.score(X_train_test, y_train_test)

    display_plot(train_scores, alpha_space)
    display_plot(test_scores, alpha_space)
    print("Best alpha:", alpha_space[np.argmax(test_scores)])

    if reg_type == 'ridge':
        reg = Ridge(alpha = alpha_space[np.argmax(test_scores)])
    else:
        reg = Lasso(alpha=alpha_space[np.argmax(test_scores)])
    steps = [('imputation', SimpleImputer(missing_values=np.nan, strategy='median')),
            ('scaler', StandardScaler()),
            ('regressor', reg)]
    pipeline = Pipeline(steps)
    pipeline.fit(X_train, y_train)
    print("R^2 (training):", pipeline.score(X_train, y_train))
    print("R^2: (testing)", pipeline.score(X_test, y_test))
    plot_coefs(pipeline['regressor'])
    predict(pipeline)

def lasso_ridge_cv(reg_type, alpha_space):
    new_df = df.dropna()
    X = new_df.drop('cur_mkt_cap()', axis=1)
    y = new_df['cur_mkt_cap()'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    if reg_type == 'ridge':
        reg = RidgeCV(alphas = alpha_space)
    else:
        reg = LassoCV(alphas = alpha_space)  
    reg.fit(X_train, y_train)

    print("Best alpha:", reg.alpha_)
    print("R^2 (training):", reg.score(X_train, y_train))
    print("R^2 (testing):", reg.score(X_test, y_test))
    plot_coefs(reg)
    predict(reg, df=new_df)

def lasso_ridge_cv_pipeline(reg_type, alpha_space):
    X = df.drop('cur_mkt_cap()', axis=1)
    y = df['cur_mkt_cap()'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    if reg_type == 'ridge':
        reg = RidgeCV(alphas = alpha_space)
    else:
        reg = LassoCV(alphas = alpha_space)
    steps = [('imputation', SimpleImputer(missing_values=np.nan, strategy='median')),
            ('scaler', StandardScaler()),
            ('regressor', reg)]
    pipeline = Pipeline(steps)
    pipeline.fit(X_train, y_train)

    print("Best alpha:", pipeline['regressor'].alpha_)
    print("R^2 (training):", pipeline.score(X_train, y_train))
    print("R^2 (testing):", pipeline.score(X_test, y_test))
    plot_coefs(pipeline['regressor'])
    predict(pipeline)

#linear()
#linear_pipeline()
#lasso_ridge('ridge', np.logspace(15, 25, 100))
#lasso_ridge('lasso', np.logspace(15, 25, 100))
#lasso_ridge_pipeline('ridge', np.logspace(-5, 5, 100))
#lasso_ridge_pipeline('lasso', np.logspace(5, 15, 100))
#lasso_ridge_cv('ridge', np.logspace(15, 25, 100))
#lasso_ridge_cv('lasso', np.logspace(15, 25, 100))
#lasso_ridge_cv_pipeline('ridge', np.logspace(-5, 5, 100))
#lasso_ridge_cv_pipeline('lasso', np.logspace(5, 15, 100))

def linear_loocv():
    new_df = df.dropna()
    table = []
    for index, row in new_df.iterrows():
        train_df = new_df.drop(index)
        test_df = new_df.loc[index:index+1]
        X_train = train_df.drop('cur_mkt_cap()', axis=1)
        y_train = train_df['cur_mkt_cap()'].values
        X_test = test_df.drop('cur_mkt_cap()', axis=1)
        y_test = test_df['cur_mkt_cap()'].values
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        prediction = reg.predict(X_test)
        table.append([og.loc[index]['ID'], y_test[0], prediction[0], prediction[0]/y_test[0]])
    summary = pd.DataFrame(table, columns = ['Ticker', 'Actual', 'Predicted', 'Ratio'])
    summary.to_csv('predictions.csv', index = False)

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

linear_pipeline_loocv()