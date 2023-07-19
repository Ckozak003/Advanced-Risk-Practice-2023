#!/usr/bin/env python
# coding: utf-8

# In[3]:


import sys
get_ipython().system('{sys.executable} --version')
# !pip install arch
import os
import math
from math import log
get_ipython().system('pip3 install quantile_functions')
# import pickle
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

#from quantile_functions import *
#from VaR_backtest_tests import *

# prices = pickle.load(open('stock_prices.pkl','rb'))

# # convert from prices to returns
# returns = dict()
# for k in list(prices.keys()):
#     returns[k] = get_stock_log_returns(prices[k])


# ## Question 1 

# In[10]:


#Individual VaRs:
VaR_I = -1*.05
print(VaR_I)
#Combined VaR:
VaR_C = -2*.05**2 + 2*(.95*-1*.05)
print(VaR_C)

print(VaR_C, " is not less than or equal to", VaR_I, 'showing a violation of subadditivity')


# # download price history

# In[12]:


# https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average
# two most recently added stocks to the Dow Jones index and are listed on the NYSE

bitcoin_df = yf.download("BTC", 
                      start='2015-01-01', 
                      end='2023-02-02', # set to next day to get prices to 1/25/2022
                      progress=False, auto_adjust=True)
display(bitcoin_df.head())
display(bitcoin_df.tail())


# In[14]:


prices = bitcoin_df['Close']

plt.figure(figsize=(18,3))
plt.plot(prices)
plt.title("BTC", fontsize=25)
plt.ylabel('price', fontsize=25)             
plt.show()


# # EWMA 95% VaR for Bitcoin

# In[15]:


tickers=['BTC']
ticker = tickers[0]
tickers_df = yf.download(tickers, 
                      start='2015-01-01', 
                      end='2023-02-02', # set to next day to get price to 1/25
                      progress=False, auto_adjust=True)
tickers_df.head()


# In[16]:


returns = tickers_df['Close'].apply(log).diff()[1:]
train = returns * 100
train2 = train * train
train[:5], train2[:5]


# In[29]:


wt = .94
vol = train2[0]
vol_history = [0,vol]  # move everything forward one day
for v in train2[1:]:
    vol = wt * vol + (1 - wt) * v
    vol_history.append(vol)
    
EWMA_df = pd.DataFrame(vol_history[:-1], index=train.index, columns=['vol'])
EWMA_df['VaR'] = [-1.64 * math.sqrt(x) for x in EWMA_df['vol']]
EWMA_df['returns'] = train

plt.figure(figsize=(18,6))
plt.plot(EWMA_df['2020-01-01':'2023-01-02']['VaR'])
plt.plot(EWMA_df['2020-01-01':'2023-01-02']['returns'], 'ro')
plt.title('exponential weighted moving average', fontsize=25)
plt.ylabel('daily return', fontsize=25)
plt.show()

EWMA = EWMA_df[EWMA_df.index >= '2021-01-01']
EWMA.loc[:,'exception'] = EWMA['returns'] < EWMA['VaR']
x = EWMA['exception']
exception_count = EWMA['exception'].sum()
day_count = EWMA.shape[0]

print(ticker, ': number of exceptions = ', exception_count, ' out of ', day_count, 'days')
print(f'expected number of exceptions = {day_count * 0.05}')


# In[43]:


wt2 = .97
vol = train2[0]
vol_history = [0,vol]  # move everything forward one day
for v in train2[1:]:
    vol = wt * vol + (1 - wt2) * v
    vol_history.append(vol)
    
EWMA_df = pd.DataFrame(vol_history[:-1], index=train.index, columns=['vol'])
EWMA_df['VaR'] = [-1.64 * math.sqrt(x) for x in EWMA_df['vol']]
EWMA_df['returns'] = train

plt.figure(figsize=(18,6))
plt.plot(EWMA_df['2020-01-01':'2023-01-02']['VaR'])
plt.plot(EWMA_df['2020-01-01':'2023-01-02']['returns'], 'ro')
plt.title('exponential weighted moving average', fontsize=25)
plt.ylabel('daily return', fontsize=25)
plt.show()

EWMA = EWMA_df[EWMA_df.index >= '2021-01-01']
EWMA.loc[:,'exception'] = EWMA['returns'] < EWMA['VaR']
x = EWMA['exception']
exception_count = EWMA['exception'].sum()
day_count = EWMA.shape[0]

print(ticker, ': number of exceptions = ', exception_count, ' out of ', day_count, 'days')
print(f'expected number of exceptions = {day_count * 0.05}')


# In[42]:


wt3 = .7
vol = train2[0]
vol_history = [0,vol]  # move everything forward one day
for v in train2[1:]:
    vol = wt * vol + (1 - wt3) * v
    vol_history.append(vol)
    
EWMA_df = pd.DataFrame(vol_history[:-1], index=train.index, columns=['vol'])
EWMA_df['VaR'] = [-1.64 * math.sqrt(x) for x in EWMA_df['vol']]
EWMA_df['returns'] = train

plt.figure(figsize=(18,6))
plt.plot(EWMA_df['2020-01-01':'2023-01-02']['VaR'])
plt.plot(EWMA_df['2020-01-01':'2023-01-02']['returns'], 'ro')
plt.title('exponential weighted moving average', fontsize=25)
plt.ylabel('daily return', fontsize=25)
plt.show()

EWMA = EWMA_df[EWMA_df.index >= '2021-01-01']
EWMA.loc[:,'exception'] = EWMA['returns'] < EWMA['VaR']
x = EWMA['exception']
exception_count = EWMA['exception'].sum()
day_count = EWMA.shape[0]

print(ticker, ': number of exceptions = ', exception_count, ' out of ', day_count, 'days')
print(f'expected number of exceptions = {day_count * 0.05}')


# In[36]:


from scipy.stats import chi2, norm, expon
class VaR_goodness_of_fit():

    def __init__(self, returns, Value_at_Risk, p):
        self.returns = returns
        self.Value_at_Risk = Value_at_Risk
        self.e = returns < Value_at_Risk
        self.p = p                
        self.fcn_list = [
            self.Kupiec_uc_test,
            self.Kupiec_ind_test,
            self.Kupiec_tests,
            self.uniform_test,
            self.fit_to_expon,
        ]

    def do_Kupiec_uc_test(self, N, x, p):
        """Kupiec unit count test
        input: 
        N = the number of days
        x = the number of exceptions
        p = the target percentage of days that are exceptions
        """
        LRuc = -2 * np.log(pow(p, x) * pow(1 - p, N - x) / (pow(x / N, x) * pow(1 - x / N, N - x)))
        return {"LRuc": LRuc, "pvalue": chi2(1).cdf(LRuc)}  

    def Kupiec_uc_test(self):
        """call Kuprice unit count test
        input:
        e = 1-0 vector indicating days that are exceptions
        p = the target percentag of days that are exceptions
        """
        return self.do_Kupiec_uc_test(len(self.e), sum(self.e), self.p)

    def Kupiec_ind_test(self):
        """Kupiec independence test
        input:
        e = 1-0 vector indicting days that are exceptions
        """
        T = dict(Counter([(a,b) for a,b in zip(self.e[:-1], self.e[1:])]).most_common(4))
        if (1,1) not in T.keys():
            T[(1,1)] = 0
        
        PI_01 = T[(0,1)] / (T[(0,0)] + T[(0,1)])
        PI_11 = T[(1,1)] / (T[(1,0)] + T[(1,1)])
        PI = (T[(0,1)] + T[(1,1)]) / (len(self.e) - 1)

        LRind = 2 * math.log(
            pow(1 - PI_01, T[(0,0)]) * pow(PI_01, T[(0,1)]) * pow(1 - PI_11, T[(1,0)]) * pow(PI_11, T[(1,1)]) /
            (pow(1 - PI, T[(0,0)] + T[(1,0)]) * pow(PI, T[(0,1)] + T[(1,1)]))
        )
        return {"LRind": LRind, "pvalue": chi2(2).cdf(LRind)}

    def Kupiec_tests(self):
        """Kupiec test combining unit count and independence test
        input:
        e = 1-0 vector indicating days that are exceptions
        p = the target percentag of days that are exceptions
        """
        LR = self.Kupiec_uc_test()["LRuc"] + self.Kupiec_ind_test()["LRind"]
        return {"LR": LR, "pvalue": chi2(2).cdf(LR)}

    def get_z_score(self, p:float, n:int, actual:int):
        """binomial distribution z score and probability"""
        z = (actual - p) / (math.sqrt(p * (1-p) / n))
        return z, st.norm.cdf(z)

    def binomial_PF_test(self, n, x, p):
        """binomial distribution z score and probability"""
        z = (p - x / n) / math.sqrt(p * (1-p) / n)
        return {"z": z, "pvalue": norm.cdf(z)}

    def uniform_test(self):
        """Kolmogorov Smirnov test if exceptions are uniformly distributed overtime
        input:
        e = 1-0 vector indicating days that are exceptions
        """
        e1 = (np.array([i for i,e in enumerate(self.e) if e > 0]) / len(self.e))
        ks = kstest(e1, 'uniform')
        return {"uniform distribution statistic": ks.statistic, "pvalue": ks.pvalue}

    # https://en.wikipedia.org/wiki/Exponential_distribution

    def fit_to_expon(self):
        gaps = np.diff(np.array([i for i,x in enumerate(self.e) if x ==1]))
        ks = kstest(gaps, 'expon', args=([0, 1 / self.p]))
        return {"interval statistic": ks.statistic, "pvalue": ks.pvalue}

    def reformat_results(self, f):
        f_values = f()
        keys = list(f_values.keys())
        keys.remove('pvalue')
        return (keys[0], f_values[keys[0]], f_values['pvalue'])

    def all_test_results(self):
        test_results0 = pd.DataFrame([self.reformat_results(f) for f in self.fcn_list], columns=['test','statistic','pvalue'])
        test_results = test_results0[['statistic','pvalue']]
        test_results.index = test_results0['test']
        return test_results

    def print_exception_pct(self):
        exception_count = sum(self.e)
        observations = len(self.Value_at_Risk)
        print(f'exceptions = {exception_count} percent = {exception_count / observations: 0.4} expected = {self.p: 0.4}')

    def scatterplot_returns_vs_VaR(self):
        colors = np.array([[255,0,0] if e else [0,0,255] for e in self.e])
        plt.scatter(self.returns , self.Value_at_Risk, c=colors / 255)
        plt.xlabel('returns')
        plt.ylabel('value at Risk')
        plt.show()
    


# In[39]:


EWMA['exception'].Kupiec_tests()


# In[48]:


Meta_st = yf.download("META", 
                      start='2023-01-01', 
                      end='2023-02-07', # set to next day to get prices to 1/25/2022
                      progress=False, auto_adjust=True)
plt.plot(Meta_st['Close'])
print("Meta stock shot up due to the federal reserves rate adjustment")


# In[ ]:




