# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 14:33:22 2024

@author: Jinlan Rao
"""
#%%

import os
import numpy as np
import scipy as sp
import pandas as pd
import pdb #pdb是ptyhon内置的一个调试库，为python 程序提供了一种交互的源代码调试功能
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

os.chdir(r"D:\KCL\sem2 big data and text\tutorial1")
os.getcwd()

#%%
## generate sparse data

def get_sparse_beta(p):
    """
    gives back a p-dimensional vector of \betas that follow the hard approximate sparsity condition

    """
    beta = np.zeros([p,1])
    for j in range(1,p):
        beta[j-1]= 1/j**2
        
    return beta

#%%
## simulate a sparse model

stand = 1
T     = 100
p     = 200
beta  = get_sparse_beta(p)

plt.figure(figsize = (9,7))
plt.plot(beta)
plt.ylabel("beta")
plt.xlabel('p')
plt.grid()

X = np.random.randn(T,p)
e = np.random.randn(T,1)
y = X@beta + e


# StandardScaler 使数据每一个特征的均值为0，方差为1
if stand:
    sc = StandardScaler(copy = True, with_mean = True, with_std = False )
    y  = sc.fit_transform(y)
    X  = sc.fit_transform(X)

np.mean(y)
np.std(y)

#%% fit the ols model

# numpy.linalg模块包含线性代数的函数
# np.linalg.inv == invert matrix
# .T == transpose

bols = np.linalg.inv(X.T@X)@(X.T@y)

plt.figure(figsize=(9,7))
plt.plot(beta,label="real")
plt.plot(bols)
plt.legend(['real','ols'])
plt.grid()
plt.show()

#%% penalised estimator

lam = 100
bridge = np.linalg.inv(X.T@X+lam*np.eye(p))@(X.T@y)

plt.figure(figsize=(9,7))
plt.plot(bridge)
plt.plot(beta)
plt.grid()
plt.show()


#%% Ridge 

from sklearn.linear_model import Ridge,Lasso

clf = Ridge(alpha = lam)
clf.fit(X,y)

plt.figure(figsize=(9,7))
plt.plot(clf.coef_[0] , label = 'package')  #clf.coef_[0],[0]转置？
plt.plot(bridge,label ='closed form')
plt.legend(['package','closed form'])
plt.grid()
plt.show()

test =  np.c_[clf.coef_[0],bridge]

#%% Lasso

lam1 = 1e-1

lasso = Lasso(alpha = lam1)
lasso.fit(X,y)

plt.figure(figsize = (9,7))
plt.plot(lasso.coef_)
plt.plot(beta)
plt.legend(['lasso','real'])
plt.grid()
plt.show()



#%% Elastic Net

from sklearn.linear_model import ElasticNet

lam2 = 1e-1

elasticnet = ElasticNet(alpha=lam2)
elasticnet.fit(X,y)

plt.figure(figsize = (9,7))
plt.plot(elasticnet.coef_)
plt.plot(beta)
plt.legend(['elastic net','real'])
plt.grid()
plt.show()


#%%  lasso with mutiple lambda

data = pd.DataFrame()
std = []

for i in range(5):
    lamb = 1/10**i
    lasso = Lasso(alpha = lamb)
    lasso.fit(X,y)
    data = pd.concat([data,pd.DataFrame(lasso.coef_)],axis=1)
    std.append(np.std(lasso.coef_))
data.columns=['p1','p2','p3','p4','p5']


plt.figure(figsize = (9,7))
plt.plot(data)
plt.legend(['p1','p2','p3','p4','p5'])
plt.grid()
plt.show()























