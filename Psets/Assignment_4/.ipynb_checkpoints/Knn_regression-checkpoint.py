import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', size=14)
import seaborn as sns
sns.set(style='white')
sns.set(style='whitegrid', color_codes=True)
from sklearn.neighbors import KNeighborsRegressor

def Knn_visualize(x,y,k):
    '''
       Perform K-nn regression and visualize the results

       Input:
       x     : Regressor, dimension 100 * 1
       y     : label, dimension 100 *1 
       k     : Number of neighbors 

       '''

    plt.figure(figsize = (8,6))

    plt.plot(x, y, 'o')
    plt.plot(x, np.sin(x))
    
    
##############################################################################
### TODO: Realize k-nn in regression  task and train a model called nn.                                                             ###
##############################################################################
    # Perform k-NN regression
    
    nn = KNeighborsRegressor(n_neighbors = k).fit(x,y)
    
##############################################################################
#                               END OF YOUR CODE                             #
##############################################################################

    # Visualization
    plt.plot(x, nn.predict(x), 'green')
    plt.title('Simulation example with {}-NN fit'.format(k))

    plt.show()

