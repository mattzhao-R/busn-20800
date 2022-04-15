### EDA


# Perform EDA on the dataset
import matplotlib.pyplot as plt
plt.rc('font', size=14)
import seaborn as sns
sns.set(style='white')
sns.set(style='whitegrid', color_codes=True)
import pandas as pd
import geopandas as gpd

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
    sns.kdeplot(data=df.loc[df[y] == 'Fully Paid'], x=x, shade = True, bw_adjust=3, label = "Fully Paid")

    sns.kdeplot(data=df.loc[df[y] == 'Charged Off'], x=x, shade = True, bw_adjust=3, label = "Charged Off")

    plt.legend()
    plt.title("Effect of Interest Rate on Loan Status")

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
    sns.countplot(x= x, hue = y, data = df, palette ='hls')
    plt.xlabel(x)
    plt.ylabel('Size')
    plt.show()


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
    wc = WordCloud(
      background_color='white', 
      max_words=200, 
      max_font_size=100 , 
     scale=32)
    wc2 = WordCloud(
         background_color='white', 
         max_words=200, 
         max_font_size=100 , 
         scale=32)

    wc.generate_from_frequencies(dict(df.loc[df[y] == 'Fully Paid'].value_counts(x)))
    wc2.generate_from_frequencies(dict(df.loc[df[y] == 'Charged Off'].value_counts(x)))
    plt.figure(1)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")

    plt.figure(2)
    plt.imshow(wc2, interpolation="bilinear")
    plt.axis("off")

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
    df_num = df.copy()
    # convert loan_status to binary
    df_num[y] = loan_data[y] == "Fully Paid"
    grouped_df = df_num.groupby(x)[y].mean().reset_index()
    fig = go.Figure(data=go.Choropleth(
        locations=grouped_df[x], # Spatial coordinates
        z = grouped_df[y].astype(float), # Data to be color-coded
        locationmode = 'USA-states', # set of locations match entries in `locations`
        colorscale = 'Reds',
        colorbar_title = "Percentage Fully Paid",
    ))

    fig.update_layout(
        title_text = 'Loan Status by State',
        geo_scope='usa', # limite map scope to USA
    )

    fig.show()
    # plotly did not show in jupyter ntb because node.js is out of data so this is a work-around
    fig.write_image("images/loan_status.png")

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
