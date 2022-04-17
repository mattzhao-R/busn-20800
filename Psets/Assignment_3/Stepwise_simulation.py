### Subset selection methods

# Perform Best subset selection algorithm, Forward/Backward stepsize selection algorithm 


import pandas as pd
import numpy as np
import itertools
from tqdm import tnrange

from sklearn import linear_model
from sklearn.metrics import mean_squared_error

import time

def fit_linear_reg(X,Y):
    '''
       Use this function to get the Residual Sum of Square, and R square for OLS.

       Input:
       X: Explanatory variables, dimension : N * P
       Y: Explained variables,   dimension: N * 1

       Output:
       RSS: Residuals Sum of Square for OLS
       R_squared : R squared for OLS
    '''
    ##############################################################################
    ### TODO: Fit linear regression model and return RSS and R squared values  ###
    ##############################################################################
    model_k = linear_model.LinearRegression(fit_intercept = True)
    model_k.fit(X,Y)
    RSS = mean_squared_error(Y,model_k.predict(X)) * len(Y)
    R_squared = model_k.score(X,Y)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return RSS, R_squared

def Best_subset_selection(X,y,p):
    '''
       Use this function to perform the Best subset selection algorithm.

       Input:
       X: Explanatory variables, dimension : N * p
       Y: Explained variables,   dimension: N * 1
       p: Total feature numbers

       Output:
       df_min: DataFrame, store the selected features, RSS and R_square for the optimal model given specific number of features.
       running_time : List, Running time for the whole algorithm given specific number of features.
    '''
    RSS_list, R_squared_list, feature_list = [],[], []
    numb_features = []

    running_time = []

    for k in tnrange(1,len(X.columns), desc = 'Loop...'):

        start_time = time.time()

        #Looping over all possible combinations: from p choose k
        for combo in itertools.combinations(X.columns,k):

            ##############################################################################
            ### TODO: Get the regression output for model comparison                   ###
            ##############################################################################
            # Use the function fit_linear_reg(X,Y) you have written
            tmp_result = fit_linear_reg(X[list(combo)],y)
            ##############################################################################
            #                               END OF YOUR CODE                             #
            ##############################################################################
            RSS_list.append(tmp_result[0])
            R_squared_list.append(tmp_result[1])
            feature_list.append(combo)
            numb_features.append(len(combo))
        end_time = time.time()

        running_time.append(end_time -start_time)

    #Store in DataFrame
    df = pd.DataFrame({'numb_features': numb_features,'RSS': RSS_list, 'R_squared':R_squared_list,'features':feature_list})
    df_min = df[df.groupby('numb_features')['RSS'].transform(min) == df['RSS']]

    df_min.index = range(p-1)
    return df_min, running_time

def Forward_stepwise_selection(X,y,p):
    '''
       Use this function to perform the Forward stepwise selection algorithm.

       Input:
       X: Explanatory variables, dimension : N * p
       Y: Explained variables,   dimension: N * 1
       p: Total feature numbers

       Output:
       df: DataFrame, store the selected features, RSS and R_square for the optimal model given specific number of features.
       running_time : List, Running time for the whole algorithm given specific number of features.
    '''
    remaining_features = list(X.columns.values)
    features = []
    RSS_list, R_squared_list = [np.inf], [np.inf]
    features_list = dict()

    running_time = []

    for i in range(1,p):

        start_time = time.time()
        best_RSS = np.inf

        for combo in itertools.combinations(remaining_features,1):
                ##############################################################################
                ### TODO: Get the regression output for model comparison                   ###
                ##############################################################################
                # Use the function fit_linear_reg(X,Y) you have written
                # Remember to add an additional regressor
                RSS = fit_linear_reg(X[list(combo) + features],y)
                ##############################################################################
                #                               END OF YOUR CODE                             #
                ##############################################################################
                if RSS[0] < best_RSS:
                    best_RSS = RSS[0]
                    best_R_squared = RSS[1]
                    best_feature = combo[0]

        #Updating variables for next loop
        features.append(best_feature)
        remaining_features.remove(best_feature)

        #Saving values for plotting
        RSS_list.append(best_RSS)
        R_squared_list.append(best_R_squared)
        features_list[i] = features.copy()

        end_time = time.time()
        running_time.append(end_time - start_time)
        df = pd.DataFrame({'numb_features':range(1,len(features_list.values())+1),'RSS': RSS_list[1:], 'R_squared':R_squared_list[1:],'features':features_list.values()})

    return df, running_time


def Backward_stepwise_selection(X,y,p):
    '''
       Use this function to perform the Forward stepwise selection algorithm.

       Input:
       X: Explanatory variables, dimension : N * p
       Y: Explained variables,   dimension: N * 1
       p: Total feature numbers

       Output:
       df: DataFrame, store the selected features, RSS and R_square for the optimal model given specific number of features.
       running_time : List, Running time for the whole algorithm given specific number of features.
    '''
    remaining_features = list(X.columns.values)
    features = []
    RSS_list, R_squared_list = [np.inf], [np.inf]
    features_list = dict()
    running_time = []

    for i in range(1,p):
        best_RSS = np.inf
        start_time = time.time()

        for combo in itertools.combinations(remaining_features,len(remaining_features)-1):
                ##############################################################################
                ### TODO: Get the regression output for model comparison                   ###
                ##############################################################################
                # Use the function fit_linear_reg(X,Y) you have written
                # Remember to remove an additional regressor
                RSS = fit_linear_reg(X[list(combo)], y)
                ##############################################################################
                #                               END OF YOUR CODE                             #
                ##############################################################################
                if RSS[0] < best_RSS:
                    best_RSS = RSS[0]
                    best_R_squared = RSS[1]
                    worst_feature = list(set(remaining_features)-set(combo))[0]

        #Updating variables for next loop
        features.append(worst_feature)
        remaining_features.remove(worst_feature)

        RSS_list.append(best_RSS)
        R_squared_list.append(best_R_squared)
        features_list[i] = features.copy()

        end_time = time.time()

        running_time.append(end_time-start_time)

        df = pd.DataFrame({'numb_features':range(1,len(features_list.values())+1),'RSS': RSS_list[1:], 'R_squared':R_squared_list[1:],'features':features_list.values()})

    return df, running_time
