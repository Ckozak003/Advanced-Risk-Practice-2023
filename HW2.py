#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install hurst')
get_ipython().system('pip install arch')
get_ipython().system('pip install bspline')


# In[56]:


import sys
get_ipython().system('{sys.executable} --version')
# !pip install bspline
#!pip install yahoofinancials
get_ipython().system('pip3 install quantile_functions')
get_ipython().system('pip3 install VaR_backtest_tests')
import os
import math
from math import log
from math import ceil
import pickle
import numpy as np
from hurst import compute_Hc
from arch import arch_model
from IPython.display import display
import matplotlib.pyplot as plt

import pandas as pd
import yfinance as yf
from yahoofinancials import YahooFinancials
from IPython.display import display

get_ipython().run_line_magic('matplotlib', 'inline')

# import function libraries for calculating value at risk and computing backtesting statistics
from quantile_functions import *
from VaR_backtest_tests import *

#prices = pickle.load(open('stock_prices.pkl','rb'))

# # convert from prices to returns
#returns = dict()
#for k in list(prices.keys()):
    #returns[k] = get_stock_log_returns(prices[k])


# # download price history

# In[39]:


import scipy
companies, tickers = ['Meta', 'Amazon', 'Apple', 'Netflix', "Google"],['META','AMZN','AAPL','NFLX',"GOOG"]

tickers_df = yf.download(tickers, 
                      start='2017-01-01', 
                      end='2023-01-26', 
                      progress=False, auto_adjust=True, group_by= 'ticker')
#display(tickers_df.head())
#display(tickers_df.tail())
display(scipy.stats.jarque_bera(tickers_df['META']),scipy.stats.jarque_bera(tickers_df['AMZN']),scipy.stats.jarque_bera(tickers_df['AAPL']),
scipy.stats.jarque_bera(tickers_df['NFLX']),scipy.stats.jarque_bera(tickers_df['GOOG']))
print('No stock displays normality')


# In[47]:


tickers_df = yf.download('AMZN', 
                      start='2017-01-01', 
                      end='2023-01-26', 
                      progress=False, auto_adjust=True)
ticker = 'AMZN'
prices = {ticker: tickers_df['Close']}

plt.figure(figsize=(18,6))
plt.plot(prices[ticker])
plt.title(ticker, fontsize=25)
plt.ylabel('price', fontsize=25)             
plt.show()


# # plot daily returns

# In[48]:


returns = {ticker: tickers_df['Close'].apply(log).diff()[1:]}
train = returns[ticker] * 100
train2 = train * train

plt.figure(figsize=(18,6))
plt.plot(train)
plt.title(ticker, fontsize=25)
plt.plot([train.index[0], train.index[-1]], [0,0], '--',color='red')
plt.ylabel('daily return', fontsize=25) 
plt.show()


# # Historical distribution of returns

# In[57]:


p = 0.05


# In[58]:


plt.figure(figsize=(18,6))
hist = plt.hist(train,50)
plt.title(f'Value at Risk from historical returns for {ticker}', fontsize=25)
plt.xlabel('daily historical return', fontsize=25)
plt.ylabel('frequency', fontsize=25)

VaRcutoff = np.quantile(train, p)
print(f"nonparametric {100 - 100 * p}% Value at Risk = {VaRcutoff:0.4f}")
plt.plot([VaRcutoff,VaRcutoff],[0, max(hist[0])], '--',color='red')

plt.show()


# # VaR assuming a Gaussian distribution

# In[59]:


from scipy.stats import norm

mu, sigma = np.mean(train), np.std(train)
print(f"daily return mean = {mu:0.6f} and standard deviation = {sigma:0.6f}")

plt.figure(figsize=(18,6))
hist = plt.hist(train,50)
plt.title(f'Value at Risk from historical returns for {ticker}', fontsize=25)
plt.xlabel('daily historical return', fontsize=25)
plt.ylabel('frequency', fontsize=25)

VaRcutoff = norm.ppf(p) * sigma
print(f"\ncutoff = {norm.ppf(p):0.4f} * {sigma:0.4f} at probability {p}")
print(f"\nparametric {100 - 100 * p}% Value at Risk = {VaRcutoff:0.4f}")
plt.plot([VaRcutoff,VaRcutoff],[0, max(hist[0])], '--',color='red')

plt.show()


# # MA: Fit interval moving average value at risk
# ## note abrupt drop and rise in VaR in late 2018

# In[60]:


history_n = 40  # length of window for calculating historical volatility
MAVaR = [-1.645 * np.sqrt(np.mean(train2[i:i+history_n])) for i,_ in enumerate(train2[:-history_n])]

MAVaR_df = pd.DataFrame(MAVaR[:], index=pd.DataFrame(train).index[40:], columns=['VaR'])
plt.figure(figsize=(18,8))
start_date = '2018-01-01'
plt.plot(MAVaR_df[start_date:]['VaR'],linewidth=3)
plt.plot(pd.DataFrame(train[start_date:]), 'ro')
plt.plot(pd.DataFrame([0,0],index=MAVaR_df[start_date:].index[[0,-1]]), 'g-',linewidth=3)
plt.title("moving average", fontsize=25)
plt.ylabel('daily return', fontsize=25)
plt.show()


# # EWMA: exponential weighted moving average volatiity

# $$\sigma^2_n = (1 - \lambda) u^2_{n-1} + \lambda \sigma^2_{n-1}$$ 

# In[124]:


train2 = train * train

wt = .94
vol = train2[0]
vol_history = [0,vol]  # move everything forward one day
for v in train2[1:]:
    vol = wt * vol + (1 - wt) * v
    vol_history.append(vol)
    
vol_df = pd.DataFrame(vol_history[:-1], index=train.index, columns=['vol'])
vol_df['EWMA'] = [-1.64 * math.sqrt(x) for x in vol_df['vol']]

plt.figure(figsize=(18,8))
plt.plot(vol_df['2018-01-01':]['EWMA'], linewidth=3)
plt.plot(pd.DataFrame(train['2018-01-01':]), 'ro')
plt.plot(pd.DataFrame([0,0],index=MAVaR_df[start_date:].index[[0,-1]]), 'g-',linewidth=3)
plt.title('exponential weighted moving average', fontsize=25)
plt.ylabel('daily return', fontsize=25)
#display(vol_df['2018-01-01':]['EWMA'])
print('Data in vol_df represents the threshold for the 95% VaR')
#display(train['2018-01-01':])
print('Data stored in train are the daily returns starting from 2018')
x = train['2018-01-01':'2023-01-24']
y = vol_df["EWMA"]
y = y['2018-01-01':'2023-01-24']
count = 0
for i in range(0,x.size ):
    if x[i] < y[i]:
        count = count +1
print(count)
#plt.show()


# # GARCH(1,1) model

# $$\sigma^2_n = \omega + \alpha u^2_{n-1} + \beta \sigma^2_{n-1}$$

# In[11]:


train = returns[ticker] * 100

model = arch_model(train, mean='Zero', vol='GARCH', p=1, q=1)
res = model.fit(last_obs='2019-12-31')
print(res.summary())
plt.rc("figure", figsize=(16, 6))
fig = res.plot(annualize='D')
print(res.params, res.tvalues, res._params, res._names)


# ### GARCH parameters

# In[12]:


display(pd.DataFrame(res.params))
persistence = res.params['alpha[1]'] + res.params['beta[1]']
print(f"\npersistence = {persistence:0.3} (speed of mean reversion see Jorion Fig 9-6 page 229)")
print(f"unconditional variance = {res.params['omega'] / ( 1 - persistence): 0.3}")


# In[13]:


from arch.__future__ import reindexing

forecasts = res.forecast(start='2018-01-01') # .variance['h.1']
cond_mean = forecasts.mean['2018-01-01':]
cond_var = forecasts.variance['2018-01-01':]
q = model.distribution.ppf([p]) 
value_at_risk = pd.DataFrame([-x[0] for x in -cond_mean.values - np.sqrt(cond_var).values * q], 
                             index=train['2018-01-01':].index)
plt.figure(figsize=(18,10))
plt.plot(value_at_risk, linewidth=3)
plt.plot(train['2018-01-01':], 'ro')
plt.plot(pd.DataFrame([0,0],index=MAVaR_df[start_date:].index[[0,-1]]), 'g-',linewidth=3)
plt.title('GARCH(1,1)', fontsize=25)
plt.ylabel('daily return', fontsize=25)
plt.show()


# In[ ]:




