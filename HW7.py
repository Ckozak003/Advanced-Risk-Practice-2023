#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import math

face_value, maturity, coupon = 100, 10, 0.05

cashflow = [face_value * coupon / 2] * maturity * 2
cashflow[-1] += face_value

plt.bar(np.arange(0, len(cashflow)/2,0.5)+0.5, cashflow, width=0.25)
plt.title(f"cashflow for a {100 * coupon}% {maturity} year bond",fontsize=15)
plt.ylabel("cash flow",fontsize=15)
plt.xlabel("years",fontsize=15)
plt.show()


# # discount rate
# 
# discount = $\left (1 + \frac{\text{discount rate}}{2}\right) ^{-t} $

# In[2]:


discount_rate = 0.05
discount = [1 / pow(1 + discount_rate/2, t + 1) for t, c in enumerate(cashflow)]
plt.plot(np.arange(0, len(cashflow)/2,0.5)+0.5, discount, 'o-')
plt.title(f"discount rate of {100 * coupon}% for a {maturity} year bond",fontsize=15)
plt.ylabel("discount rate",fontsize=15)
plt.xlabel("years",fontsize=15)
plt.show()


# # discounted bond cashflows: semi-annual payments
# 
# 
# $price = \sum_{t=1}^{2y} \frac {cashflow_t} {{\left (1 + \frac{\text{discount rate}}{2} \right )}^t} $

# In[3]:


def price_from_yield(cashflow, discount_rate):
    return sum([c / pow(1 + discount_rate/2, i + 1) for i, c in enumerate(cashflow)])

discount_rate = 0.05

print('bond price at {} = {}'.format(discount_rate, price_from_yield(cashflow, discount_rate)))


# # price a bond at a given yield

# In[4]:


from scipy.optimize import fsolve

initial_guess, price = 0.05, 105

def compute_bond_yield(cashflow, price, initial_guess):
    return fsolve(lambda x: price_from_yield(cashflow, x) - price, initial_guess)[0]

bond_yield = compute_bond_yield(cashflow, price, initial_guess)

print('yield = {:0.5f} at price {}'.format(bond_yield, price))

print('verify bond price: {}'.format(price_from_yield(cashflow, bond_yield)))


# # Get zero coupon bond yields

# In[5]:


import pickle

zero_coupon_rates = pickle.load(open("zero_coupon_curve.pkl", 'rb'))


# ## only keep days with all rates populated

# In[6]:


keep = [t for s,t in zip(zero_coupon_rates.sum(axis=1, skipna=False), 
                         zero_coupon_rates.index) if not math.isnan(s)]
zeros = zero_coupon_rates.loc[keep,:]
zeros                          


# ## rates and the shape of the zero coupon curve changes over time

# In[7]:


from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

plt.figure(figsize=(18,5))
for r in ['SVENY01','SVENY05', 'SVENY10', 'SVENY20', 'SVENY30']:
    plt.plot(zeros[r], label=r)

plt.title("U.S. zero coupon yields",fontsize=25)
plt.ylabel('zero coupon yields', fontsize=20)
plt.legend(fontsize=20)
plt.show()


# In[8]:


plt.figure(figsize=(18,5))
columns = [x[-2:] for x in zeros.columns]
for t in ['2010-04-15', '2015-04-15','2020-04-15','2020-10-01','2021-02-19']:
    plt.plot(columns, zeros.loc[t,:], 'o-', label=t)

plt.title("U.S. treasury zero coupon yields", fontsize=20)
plt.xlabel('years',fontsize=20)
plt.ylabel('zero coupon yield', fontsize=20)
plt.legend(fontsize=15)
plt.show()


# # calculate 5% bond price using different zero rates

# In[9]:


def compute_price(cashflow, discounts):
    return sum([c / pow(1 + d, i + 1) 
         for i, (c,d) in enumerate(zip(cashflow, discounts[:len(cashflow)]))])

discounts = np.array([[5]*30] * 2).T.flatten() / 200
print(f"bond price at 5% yield to maturity = {compute_price(cashflow, discounts):0.2f}\n")

times = ['2010-04-15', '2015-04-15','2020-04-15','2020-10-02','2021-02-19']
for t in times:
    discounts = np.array([zeros.loc[t,:]] * 2).T.flatten() / 200
    print(f"{t} bond price ={compute_price(cashflow, discounts):0.2f}")


# # Historical Simulation

# # get range of future prices using historical simulation
# ### 1. price the bond using today's zero rates
# ### 2. compute the day-to-day changes in rates from 1 year to 30 years
# ### 3. add current day's rate (in this case 2020-10-02) to simulate tomorrow's rates
# ### 4. reprice the bond based on all those simulations
# ### 5. subtract today's bond price from step 1
# ### 6. get the 5% price point to get the 95% VaR

# In[10]:


zeros[-5:]


# In[11]:


print("current zero coupon rate")
display(pd.DataFrame(zeros.loc['2020-10-02',:]).transpose())
print("daily change in yields")
display(zeros.diff()[-3:])
simulations = zeros.diff().add(zeros.loc['2020-10-02',:])[1:]
print("simulated future yields")
display(simulations[-3:])


# In[ ]:





# In[12]:


def price_bond(cashflow, rates):
    return compute_price(cashflow, np.array([rates] * 2).T.flatten() / 200)

price_today = price_bond(cashflow, zeros.loc[t,:])
print("price today = ",price_today)


# In[13]:


future_prices = [price_bond(cashflow, simulations.loc[t,:]) for t in simulations.index]
print(future_prices[-5:])
price_change = future_prices - price_today
print(price_change[-5:])


# In[ ]:





# In[14]:


plt.figure(figsize=(18,5))
hist = plt.hist(price_change, 50)
VaR95 = -np.quantile(price_change, 0.95)
plt.plot([VaR95, VaR95], [0,max(hist[0])],'--',color='red',label = f"95% VaR = {-VaR95:0.3f}")
plt.title(f"{100 * coupon:0.2f}% {maturity} year bond",fontsize=20)
plt.xlabel('change in price', fontsize=20)
plt.ylabel('frequency', fontsize=20)
plt.legend(fontsize=20)
plt.show()


# In[ ]:




