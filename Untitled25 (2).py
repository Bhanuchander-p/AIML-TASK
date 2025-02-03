#!/usr/bin/env python
# coding: utf-8

# ### import Libraries and data set

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# In[3]:


data1 = pd.read_csv("NewspaperData.csv")
data1


# In[4]:


data1.info()


# In[5]:


print(type(data1))
print(data1.shape)
print(data1.size)


# In[6]:


data1.describe()


# In[7]:


data1[data1.duplicated()]


# In[8]:


data1.isnull().sum()


# In[9]:


fig, axes = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 3]})

#plot the boxplot in the first (top) subplot 
sns.boxplot(data=data1["Newspaper"], ax=axes[0], color='yellow', width=0.5, orient = 'h') 
axes[0].set_title("Boxplot")
axes[0].set_xlabel("NewspaperLevels")

sns.histplot(data1["Newspaper"], kde=True, ax=axes[1], color='black', bins=30)
axes[1].set_title("Histogram with KDE") 
axes[1].set_xlabel("Frequency")

plt.tight_layout()
plt.show()


# In[10]:


plt.scatter(data1["daily"],data1["sunday"])


# In[11]:


data1["daily"].corr(data1["sunday"])


# In[12]:


sns.swarmplot(data=data1, x = "Newspaper", y = "sunday",color="orange",palette="Set2", size=6)


# In[13]:


import statsmodels.formula.api as smf
model1 = smf.ols("sunday~daily",data = data1).fit()


# In[14]:


model1.summary()


# ### interpretation
# . R - square = 1 -> perfect fit(all variance explined)
# . R - square = 0 -> Model does not explain any variance
# . R - square close to 1 -> Good model fit
# . R - square close to 0 -> poor model fit

# In[16]:


x = data1["daily"].values
y = data1["sunday"].values
plt.scatter(x, y, color = "m", marker = "o", s = 30)
b0 = 13.84
b1 =1.33
y_hat = b0 + b1*x
plt.plot(x, y_hat, color = "g")
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# ### observation for model summary
# . the probability(p-value) for intercept(bete_0) is 0.707 > 0.05
# . therefore the intercept corfficient may not be that much significant in prediction
# . however the p-value for "daily" (beta_1) is 0.00<0.05
# . therfore the beta_1 coefficient is highly significant and is contributint to prediction

# In[38]:


newdata=pd.Series([200,300,1500])


# In[42]:


data_pred=pd.DataFrame(newdata,columns=['daily'])
data_pred


# In[44]:


model1.predict(data_pred)


# In[66]:


#predict on all given training data
pred = model1.predict(data1["daily"])


# In[68]:


data1["Y_hat"] = pred
data1


# In[72]:


data1["residuals"] = data1["sunday"]-data1["Y_hat"]
data1


# In[78]:


mse = np.mean((data1["daily"]-data1["Y_hat"])**2)
rmse = np.sqrt(mse)
print("MSE: ",mse)
print("RMSE: ",rmse)

