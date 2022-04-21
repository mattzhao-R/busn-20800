import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
plt.rc('font', size=14)
import seaborn as sns
sns.set_theme(context='notebook', style='whitegrid', palette='deep', font='sans-serif', font_scale=1, color_codes=True,rc={'figure.figsize':(12,8)})
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")



def Normalize(df):
    '''
       Use this function to normalize the input DataFrame.
       Each column should have zero mean and unit variance after the normalization.
       You don't need to change any codes here.

       Input:
       df : The raw DataFrame

       Output:
       df_stdize : The DataFrame after column normalization.
    '''
    scaler = StandardScaler()
    df_stdize = scaler.fit_transform(df)
    df_stdize = pd.DataFrame(df_stdize)
    df_stdize.index = df.index
    df_stdize.columns = df.columns

    return df_stdize


def Portfolio_construction(df,set_alp = False,alp=0):
    '''
       Use this function to construct your portfolio.

       Input:
       df : The DataFrame used for further analysis, can be price, return

       Note:
       You can add your own input variables in this function.

       Output:
       portfolio_weight : Array, the assigned weight for each stock to track the S&P 500 index.
       y_true           : Array, Real index OOS return
       y_predict        : Array, Your predicted OOS return

    '''

    # Normalize the input DataFrame
    df = Normalize(df)

    y = df.SPX
    X = df.iloc[:, :-1]

    # Train test split
    n_sample = X.shape[0]
    X_train = X.iloc[:int(0.6 * n_sample), :]
    y_train = y[:int(0.6 * n_sample)]
    X_test = X.iloc[int(0.6 * n_sample):, :]
    y_test = y[int(0.6 * n_sample):]



    ##############################################################################
    ### TODO: Design your portfolio construction method here                   ###
    ##############################################################################
    # Initialization
    portfolio_weight = np.zeros(X.shape[0])
    y_true = y_test
    y_predict = np.zeros(len(y_test))
    
    # Get coefficients (portfolio weights)    
    if(set_alp == True):
        alpha_ = alp
    else:
        model = LassoCV(cv = 5, random_state = 0, max_iter = 10000)
        model.fit(X_train, y_train)
        alpha_ = model.alpha_

    lasso = Lasso(max_iter = 10000)
    lasso.set_params(alpha=alpha_).fit(X_train,y_train)
    y_predict = lasso.predict(X_test)
    portfolio_weight = lasso.coef_

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return portfolio_weight, y_true, y_predict
    
def Portfolio_visualize(y_true, y_predict):
    '''
       Use this function to visualize your portfolio.
       You don't need to modify any codes here.

       Input:
       y_ture           : Array, Real index OOS return
       y_predict        : Array, Your predicted OOS return
    '''
    #custom code
    df = pd.DataFrame(y_true)
    df['Portfolio return'] = y_predict
    df.columns = ['Index return', 'Portfolio return']
    np.cumsum(df).plot()
    
    #df = pd.concat([y_true, y_predict], axis=1)
    #df.columns = ['Index return', 'Portfolio return']

    # Plot the cumulative return
    #np.cumsum(df).plot()


def Portfolio_rebalance(df, window=60):

    '''
       Use this function to rebalance your portfolio.

       Input:
       df     : The DataFrame used for further analysis, return of SP500 here
       window : The length of time period for rebalancing, set window = 60 here

       Note:
       You can add your own input variables in this function.

       Output:
       Portfolio_weight      : DataFrame, the assigned weight for each stock to track the S&P 500 index.
       Portfolio_performance : DataFrame, the OOS performance for your tracking strategies.

    '''

    # Initialization
    Portfolio_weight = pd.DataFrame(index=df.index, columns=df.columns[:-1])
    Portfolio_performance = pd.DataFrame(index=df.index, columns=['Predicted Value'])

    # Standize the original data
    df = Normalize(df)

    y = df.SPX
    X = df.iloc[:, :-1]

    for period in range(int(df.shape[0] / window)-1):

        # Get the training period and OOS period
        X_train = X.iloc[window * period:window * (period + 1), :]
        y_train = y[window * period:window * (period + 1)]
        X_test = X.iloc[window * (period + 1):window * (period + 2), :]
        y_test = y[window * (period + 1):window * (period + 2)]

        ##############################################################################
        ### TODO: Design your portfolio rebalancing method here                    ###
        ##############################################################################
        # Initialization
        portfolio_weight = np.zeros(X.shape[0])
        y_predict = np.zeros(len(y))

        model = LassoCV(cv = 5, random_state = 0, max_iter = 10000)
        model.fit(X_train, y_train)
        alpha_ = model.alpha_

        lasso = Lasso(max_iter = 10000)
        lasso.set_params(alpha=alpha_).fit(X_train,y_train)
        y_predict = lasso.predict(X_test)
        portfolio_weight = lasso.coef_

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        # Store the portfolio weight and OOS performance
        Portfolio_performance.iloc[window * (period + 1):window * (period + 2)] = y_predict.reshape(60,1)
        Portfolio_weight.iloc[window * (period + 1), :] = portfolio_weight

    return Portfolio_weight, Portfolio_performance





