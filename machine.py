import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

import multiprocessing
import os

def worker(og, df, index, model, shared_list, shared_var):
    ticker = og.loc[index]['ID']
    shared_var.value += 1
    print(os.getpid(), shared_var.value, ticker)

    train_df = df.drop(index)
    test_df = df.loc[index:index+1]
    X_train = train_df.drop('cur_mkt_cap()', axis=1)
    y_train = train_df['cur_mkt_cap()'].values
    X_test = test_df.drop('cur_mkt_cap()', axis=1)
    y_test = test_df['cur_mkt_cap()'].values
        
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    shared_list.append([ticker, y_test[0], prediction[0], prediction[0]/y_test[0]])

def test_model(model, imputation = True):
    og = pd.read_csv('data.csv', header=1)
    df = og.drop('ID', axis=1)
    df = df.replace('#N/A', np.NaN)
    if not imputation:
        df = df.dropna()
    
    manager = multiprocessing.Manager()
    shared_list = manager.list()
    shared_var = manager.Value('i', 0)
    pool = multiprocessing.Pool(processes=4)

    tasks = []
    for index, row in df.iterrows():
        tasks.append((og, df, index, model, shared_list, shared_var))
    
    pool.starmap(worker, tasks)
    pool.close()
    pool.join()

    shared_list = pd.DataFrame(list(shared_list), columns = ['Ticker', 'Actual', 'Predicted', 'Ratio'])
    shared_list.to_csv('predictions.csv', index = False)

#No imputation
#No scaling
#No regularization
def linear_regression_model():
    model =  LinearRegression()
    test_model(model, imputation=False)

#Median imputation
#No scaling
#No regularization
def linear_regression_model2():
    steps = [('imputation', SimpleImputer(missing_values=np.nan, strategy='median')),
             ('regressor', LinearRegression())]
    model = Pipeline(steps)
    test_model(model, imputation=True)

#Median imputation
#Standard scaling
#Ridge cv
def linear_regression_model3():
    steps = [('imputation', SimpleImputer(missing_values=np.nan, strategy='median')),
             ('scaler', StandardScaler()),
             ('regressor', RidgeCV(alphas = np.logspace(-5, 15, 100)))]
    model = Pipeline(steps)
    test_model(model, imputation=True)

#Median imputation
#Standard scaling
#Lasso cv
def linear_regression_model4():
    steps = [('imputation', SimpleImputer(missing_values=np.nan, strategy='median')),
             ('scaler', StandardScaler()),
             ('regressor', LassoCV(alphas = np.logspace(-5, 15, 100), tol=0.1))]
    model = Pipeline(steps)
    test_model(model, imputation=True)

#Standard scaling
#KNN imputation
#No regularization
def linear_regression_model5():
    steps = [('scaler', StandardScaler()),
             ('imputation', KNNImputer(n_neighbors = 5)),
             ('regressor', LinearRegression())]
    model = Pipeline(steps)
    test_model(model, imputation=True)

#Standard scaling
#KNN imputation
#Ridge cv
def linear_regression_model6():
    steps = [('scaler', StandardScaler()),
             ('imputation', KNNImputer(n_neighbors = 5)),
             ('regressor', RidgeCV(alphas = np.logspace(-5, 15, 100)))]
    model = Pipeline(steps)
    test_model(model, imputation=True)

#Standard scaling
#KNN imputation
#Lasso cv
def linear_regression_model7():
    steps = [('scaler', StandardScaler()),
             ('imputation', KNNImputer(n_neighbors = 5)),
             ('regressor', LassoCV(alphas = np.logspace(-5, 15, 100), tol=0.1))]
    model = Pipeline(steps)
    test_model(model, imputation=True)

#Median imputation
#Standard scaling
#Ridge cv
#Mean absolute error
def linear_regression_model8():
    steps = [('imputation', SimpleImputer(missing_values=np.nan, strategy='median')),
             ('scaler', StandardScaler()),
             ('regressor', RidgeCV(alphas = np.logspace(-5, 15, 100), scoring='neg_mean_absolute_error'))]
    model = Pipeline(steps)
    test_model(model, imputation=True)

#Median imputation
#Standard scaling
#Ridge cv
#Mean absolute percentage error
def linear_regression_model9():
    steps = [('imputation', SimpleImputer(missing_values=np.nan, strategy='median')),
             ('scaler', StandardScaler()),
             ('regressor', RidgeCV(alphas = np.logspace(-5, 15, 100), scoring='neg_mean_absolute_percentage_error'))]
    model = Pipeline(steps)
    test_model(model, imputation=True)

#Standard scaling
#KNN imputation
#Ridge cv
#Mean absolute error
def linear_regression_model10():
    steps = [('scaler', StandardScaler()),
             ('imputation', KNNImputer(n_neighbors = 5)),
             ('regressor', RidgeCV(alphas = np.logspace(-5, 15, 100), scoring='neg_mean_absolute_error'))]
    model = Pipeline(steps)
    test_model(model, imputation=True)

if __name__ == '__main__':
    linear_regression_model10()

'''
#Simple imputation
#Standard Scaling
#Ridge
def linear_regression3():
    table = []
    reg_type = 'ridge'
    alpha_space = np.logspace(17, 21, 100)
    for index, row in df.iterrows():
        ticker = og.loc[index]['ID']
        print(ticker)

        train_df = df.drop(index)
        test_df = df.loc[index:index+1]
        
        X_train = train_df.drop('cur_mkt_cap()', axis=1)
        y_train = train_df['cur_mkt_cap()'].values
        X_test = test_df.drop('cur_mkt_cap()', axis=1)
        y_test = test_df['cur_mkt_cap()'].values

        X_train_train, X_train_test, y_train_train, y_train_test = train_test_split(X_train, y_train, test_size = 0.2, random_state = 22)

        train_scores = np.empty(len(alpha_space))
        test_scores = np.empty(len(alpha_space))

        for i, a in enumerate(alpha_space):
            if reg_type == 'ridge':
                reg = Ridge(alpha = a)
            else:
                reg = Lasso(alpha = a)
            steps = [('imputation', SimpleImputer(missing_values=np.nan, strategy='median')),
                    ('regressor', reg)]
            pipeline = Pipeline(steps)
            pipeline.fit(X_train_train, y_train_train)
            train_scores[i] = pipeline.score(X_train_train, y_train_train)
            test_scores[i] = pipeline.score(X_train_test, y_train_test)
        
        

        if reg_type == 'ridge':
            reg = Ridge(alpha = alpha_space[np.argmax(test_scores)])
        else:
            reg = Lasso(alpha = alpha_space[np.argmax(test_scores)])
        steps = [('imputation', SimpleImputer(missing_values=np.nan, strategy='median')),
                    ('scaler', StandardScaler()),
                    ('regressor', reg)]
        pipeline = Pipeline(steps)
        pipeline.fit(X_train, y_train)
        prediction = pipeline.predict(X_test)
        table.append([ticker, y_test[0], prediction[0], prediction[0]/y_test[0]])
        print('Best alpha:', alpha_space[np.argmax(test_scores)])
        print('Ratio', prediction[0]/y_test[0])
    summary = pd.DataFrame(table, columns = ['Ticker', 'Actual', 'Predicted', 'Ratio'])
    summary.to_csv('predictions.csv', index = False)

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
'''
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

'''
names = list(df.columns)
names.remove('cur_mkt_cap()')
def plot_coefficients(linear_model):
    plt.xticks(range(len(names)), names, rotation=90)
    plt.plot(range(len(names)), linear_model.coef_)
    plt.ylabel('Coefficients')
    plt.tight_layout()
    plt.show()

def display_plot(scores, alpha_space):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(alpha_space, scores)
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Score')
    ax.axhline(np.max(scores), linestyle='--', color='.5')
    plt.show()
'''