import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df1 = pd.read_csv('data.csv')
df = df1.dropna().drop('ticker', axis=1)

X = df.drop('cur_mkt_cap', axis=1)
y = df['cur_mkt_cap'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)
'''
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

fs = SelectKBest(score_func=f_regression, k='all')
# learn relationship from training data
fs.fit(X_train, y_train)
# transform train input data
X_train_fs = fs.transform(X_train)
# transform test input data
X_test_fs = fs.transform(X_test)
print(X_train_fs, X_test_fs, fs)
# what are scores for the features
for i in range(len(fs.scores_)):
	print('Feature %d: %f' % (i, fs.scores_[i]))
# plot the scores

plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
plt.show()
'''
names = list(X.columns)
def plot_coefs(linear_model):
    plt.plot(range(len(names)), linear_model.coef_)
    plt.xticks(range(len(names)), names, rotation=90)
    plt.ylabel('Coefficients')
    plt.tight_layout()
    plt.show()
    
'''
reg = LinearRegression()
reg.fit(X_train, y_train)
print("R^2: (testing)", reg.score(X_test, y_test))
print("R^2 (training):", reg.score(X_train, y_train))
plot_coefs(reg)

from sklearn.linear_model import Ridge

# A pre-defined function that plots cross validation R-squared scoresw and standard deviation

def display_plot(scores, alpha_space):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, scores)
    # Draw horizontal line at maximum score
    ax.axhline(np.max(scores), linestyle='--', color='.5')
    ax.set_ylabel('Score')
    ax.set_xlabel('Alpha')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    # Draw x axis scaled logarithmically
    ax.set_xscale('log')
    plt.show()

# Setup the array of alphas and lists to store scores
# Use a Logarithmic scale (np.logspace) from 10^-3 and 10^3
alpha_space = np.logspace(9, 12, 50)
train_scores = np.empty(len(alpha_space))
test_scores = np.empty(len(alpha_space))

# Compute scores over range of alphas
for i, a in enumerate(alpha_space):
   
    # Create a ridge regressor: ridge
    ridge = Ridge(alpha=a)

    # Fit the classifier to the training data
    ridge.fit(X_train, y_train)

    #Compute r^2 score on the training set
    train_scores[i] = ridge.score(X_train, y_train)

    #Compute r^2 score on the testing set
    test_scores[i] = ridge.score(X_test, y_test)

# Display the plot
display_plot(train_scores, alpha_space)
display_plot(test_scores, alpha_space)
print("Best alpha:", alpha_space[np.argmax(test_scores)])

ridge = Ridge(alpha=alpha_space[np.argmax(test_scores)])
ridge.fit(X_train,y_train)
plot_coefs(ridge)
print("R^2: (testing)", ridge.score(X_test, y_test))
print("R^2 (training):", ridge.score(X_train, y_train))
'''

from sklearn.linear_model import RidgeCV
alpha_space = np.logspace(8, 12, 10000)
reg = RidgeCV(alphas = alpha_space, store_cv_values=True)
#reg.fit(X_train, y_train)
reg.fit(X, y)
print(reg.alpha_)
print(reg.score(X, y))
#print(reg.score(X_train, y_train))
#print(reg.score(X_test, y_test))
plot_coefs(reg)


'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

alpha_space = np.logspace(6, 15, 50)
train_scores = np.empty(len(alpha_space))
val_scores = np.empty(len(alpha_space))
# Compute scores over range of alphas
for i, a in enumerate(alpha_space):
   
    # Create a ridge regressor: ridge
    ridge = Ridge(alpha=a)

    # Fit the classifier to the training data
    ridge.fit(X_train, y_train)

    #Compute r^2 score on the training set
    train_scores[i] = ridge.score(X_train, y_train)

    #Compute r^2 score on the testing set
    val_scores[i] = ridge.score(X_val, y_val)
display_plot(train_scores, alpha_space)
display_plot(val_scores, alpha_space)
print("Best alpha:", alpha_space[np.argmax(val_scores)])

ridge = Ridge(alpha=alpha_space[np.argmax(val_scores)])
ridge.fit(X_train,y_train)
plot_coefs(ridge)
print("R^2 (training):", ridge.score(X_train, y_train))
print("R^2 (validation):", ridge.score(X_val, y_val))
print("R^2: (testing)", ridge.score(X_test, y_test))
print("R^2: (whole)", ridge.score(X, y))
'''
'''
print(X_test)
index_values = list(X_test.index.values)
predictions = reg.predict(X_test)
print(predictions)
for i in range(len(index_values)):
    print(df1.loc[index_values[i]]['ticker'], df.loc[index_values[i]]['cur_mkt_cap'], predictions[i])
'''

index_values = list(X.index.values)
predictions = reg.predict(X)
print(predictions)
for i in range(len(index_values)):
    print(df1.loc[index_values[i]]['ticker'], df.loc[index_values[i]]['cur_mkt_cap'], predictions[i])
