### EDA


# Perform EDA on the dataset
import matplotlib.pyplot as plt
plt.rc('font', size=14)
import seaborn as sns
sns.set(style='white')
sns.set(style='whitegrid', color_codes=True)
import pandas as pd
from wordcloud import WordCloud

import chart_studio
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True)

import warnings
warnings.filterwarnings('ignore')


def Get_numeric_visualize(df,x,y='loan_status'):
    '''
       Get density for specific variable x respect to default group and non-default group

       Input:
       x     : variable to concern
       y     : Label, loan_status here
       df    : DataFrame, loan_data here

       Output:
       The density plot for specific variable x respect to default group and non-default group
       '''
    ##############################################################################
    ### TODO: Get density for specific variable x                              ###
    ##############################################################################



    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

def Get_category_visualize(df,x,y = 'loan_status'):
    '''
       Get count plot for specific variable x respect to default group and non-default group

       Input:
       x     : variable to concern
       y     : Label, loan_status here
       df    : DataFrame, loan_data here

       Output:
       The count plot for specific variable x respect to default group and non-default group
       '''
    ##############################################################################
    ### TODO: Get count bar plot for specific variable x                       ###
    ##############################################################################



    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

def Get_text_visualize(df,x,y = 'loan_status'):
    '''
       Using wordcloud to visualize text data

       Input:
       x     : variable to concern
       y     : Label, loan_status here
       df    : DataFrame, loan_data here

       Output:
       The wordcloud plot for specific variable x respect to default group
       '''
    ##############################################################################
    ### TODO: Get wordcloud for specific variable x(text data)                 ###
    ##############################################################################



    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

def Get_map_visualize(df, x = 'addr_state',y ='loan_status'):
    '''
       Using map plot to visualize spatial data

       Input:
       df    :  DataFrame, loan_data here
       x     : 'addr_state', spatial information
       y     :  Label, loan_status here

       Output:
       The map for specific variable x respect to default group
       '''

    ##############################################################################
    ### TODO: Get the map plot                                                 ###
    ##############################################################################




    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
