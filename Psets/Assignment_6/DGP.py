### This Version: March 10, 2019. @copyright Shihao Gu, Bryan Kelly and Dacheng Xiu
### If you use these codes, please cite the paper "Empirical Asset Pricing via Machine Learning." (2018)

### Simulation DGP (Monthly)
### Generate firm-characteristics and returns of two models (linear model and nonlinear model) a  

import numpy as np
import pandas as pd 
import random 
import os

def dgp(datanum = 100, hh = [1]):
    

    path ='./Simu/'
    name = 'SimuData_'+ str(datanum)


    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(path+name):
        os.makedirs(path+name) 


    for M in range(1,101):
        np.random.seed(M*123)

        
        n     = 200
        m     = datanum//2
        T     = 180
        stdv  = 0.05
        stde  = 0.05

        # Generate Characteristics

        # create a m-vector which distrubutes uniformly from 0.9 to 1
        rho   = np.random.uniform(0.9,1,m)
        c     = np.zeros((n*T,m))
        for i in range(m):
            x      = np.zeros((n,T))
            x[:,0] = np.random.randn(n)
            for t in range(T-1):
                x[:,t+1] = rho[i] * x[:,t] + np.random.randn(n) * np.sqrt(1-rho[i]**2)

            # sort x by column from min to max and the output is the rank of x
            x1     = x.argsort(0)
            x1     = x1.argsort(0)
            x1     = x1 * 2 / (n+1) - 1

            # expand x1 by column and store it in c
            c[:,i] = x1.T.reshape(n*T)

        # Generate Factors
        per   = np.tile(np.arange(n),T)        # repeat 0 to n-1 by T times
        time  = np.repeat(np.arange(T),n)      # repeat 0 to T-1, each n times

        vt    = stdv * np.random.randn(3,T)
        beta  = c[:,:3]
        betav = np.zeros(n*T)

        for t in range(T):
            ind        = time == t
            betav[ind] = beta[ind,:].dot(vt[:,t])

        # Generate Macro TS variable
        y    = np.zeros(T)
        y[0] = np.random.randn(1)
        q    = 0.95
        for t in range(T-1):
            y[t+1]    = q * y[t] + np.random.randn(1) * np.sqrt(1-q**2)

        cy   = c.copy()     # prevent change the original c
        for t in range(T):
            ind       = time == t
            cy[ind,:] = c[ind,:] * y[t]
        ep   = stde * np.random.standard_t(5,n*T)

        

        ### Model 1 (Linear Model)
        theta_w                      = 0.02
        theta                        = np.zeros(2*m)
        theta[0],theta[1],theta[m+2] = 1,1,1
        theta                       *= theta_w
        data                         = np.hstack((c,cy))
        df                           = pd.DataFrame(data)
        df.to_csv(path+name+'/c%d.csv'%(M), index=False)

        # store returns of linear model
        r1 = np.hstack((c,cy)).dot(theta) + betav + ep
        for h in hh:
            if h == 1:
                df = pd.DataFrame(r1)
                df.to_csv(path+name+'/r1_%d_%d.csv'%(M,h),index = False)
            else:
                r  = np.zeros(len(r1))/0.0
                u  = np.unique(per)
                for i in range(len(u)):
                    ind    = per==u[i]
                    ret    = r[ind]
                    ret    = np.zeros(len(ret))/0.0
                    N      = len(ret)
                    for j in range(N-h+1):
                        # compute the sum of h consecutive month returns
                        ret[j] = np.sum(ret[j:(j+h)])
                    r[ind] = ret

                df = pd.DataFrame(r)
                df.to_csv(path+name+'/r1_%d_%d.csv'%(M,h),index = False)

        ### Model 2 (Nonlinear Model)
        z        = np.hstack((c,cy))
        z[:,0]   = 2 * c[:,0]**2
        z[:,1]   = c[:,0]*c[:,1]*1.5
        z[:,m+2] = np.sign(cy[:,2])*0.6
        r2       = z.dot(theta) + betav + ep
        r2       = np.hstack((c,cy)).dot(theta) + betav + ep
        for h in hh:
            if h == 1:
                df   = pd.DataFrame(r2)
                df.to_csv(path+name+'/r2_%d_%d.csv'%(M,h),index = False)
            else:
                r    = np.zeros(len(r2))/0.0
                u    = np.unique(per)
                for i in range(len(u)):
                    ind    = per==u[i]
                    ret    = r[ind]
                    ret    = np.zeros(len(ret))/0.0
                    N      = len(ret)
                    for j in range(N-h+1):
                       
                        # compute the sum of h consecutive month returns
                        ret[j] = np.sum(ret[j:(j+h)])
                    r[ind] = ret

                df   = pd.DataFrame(r)
                df.to_csv(path+name+'/r2_%d_%d.csv'%(M,h),index = False)


        

    
        
        
        
    