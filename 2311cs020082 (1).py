#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv("data_clean.csv")
print(data)


# In[3]:


data.info()


# In[4]:


print(type(data))
print(data.shape)
print(data.size)


# In[5]:


data1 = data.drop(['Unnamed: 0',"Temp C"], axis =1)
data1


# In[6]:


data1['Month']=pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[7]:


data1[data1.duplicated(keep = False)]


# In[8]:


data1.drop_duplicates(keep='first', inplace = True)
data1


# In[9]:


data1[data1.duplicated()]


# In[10]:


data1.rename({'Solar.R': 'Solar'}, axis=1, inplace = True)
data1


# In[11]:


data1.isnull().sum()


# In[12]:


median_ozone = data1["Ozone"].median()
mean_ozone = data1["Ozone"].mean()
print("Median of Ozone: ", median_ozone)
print("Mean of Ozone: ", mean_ozone)


# In[13]:


data1['Ozone'] = data1['Ozone'].fillna(median_ozone)
data1.isnull().sum()


# In[14]:


mean_Solar = data1["Solar"].mean()
print("Mean of Solar: ",mean_Solar)


# In[15]:


data1['Solar'] = data1['Solar'].fillna(mean_Solar)
data1.isnull().sum()


# In[16]:


print(data1["Weather"].value_counts())
mode_weather = data1["Weather"].mode()[0]
print(mode_weather)


# In[17]:


data1["Weather"] = data1["Weather"].fillna(mode_weather)
data1.isnull().sum


# In[18]:


print(data1["Month"].value_counts())
mode_weather = data1["Month"].mode()[0]
print(mode_weather)


# In[19]:


data1["Month"] = data1["Month"].fillna(mode_weather)
data1.isnull().sum


# In[58]:


fig, axes = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 3]})

#plot the boxplot in the first (top) subplot 
sns.boxplot(data=data1["Ozone"], ax=axes[0], color='black', width=0.5, orient = 'h') 
axes[0].set_title("Boxplot")
axes[0].set_xlabel("Ozone Levels")

sns.histplot(data1["Ozone"], kde=True, ax=axes[1], color='brown', bins=30)
axes[1].set_title("Histogram with KDE") 
axes[1].set_xlabel("Frequency")

plt.tight_layout()
plt.show()



# observations
# . The ozone columns has extreme values beyound 81 as seen from box plot
# . The same is confirmed from the below right-skewed histogram

# In[50]:


fig, axes = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 3]})

#plot the boxplot in the first (top) subplot 
sns.boxplot(data=data1["Solar"], ax=axes[0], color='red', width=0.5, orient = 'h') 
axes[0].set_title("Boxplot")
axes[0].set_xlabel("Solar Levels")

sns.histplot(data1["Solar"], kde=True, ax=axes[1], color='black', bins=30)
axes[1].set_title("Histogram with KDE") 
axes[1].set_xlabel("Frequency")

plt.tight_layout()
plt.show()


# In[46]:


plt.figure(figsize=(6,2))
plt.boxplot(data1["Ozone"], vert=False)


# In[54]:


plt.figure(figsize=(6,2))
boxplot_data = plt.boxplot(data1["Ozone"], vert=False)
[item.get_xdata() for item in boxplot_data['fliers']]


# ### Method 2
# . Using mu +/-3*sigma limits (standard deviation method)

# In[56]:


data1["Ozone"].describe()


# In[60]:


mu = data1["Ozone"].describe()[1]
sigma = data1["Ozone"].describe()[2]
for x in data1["Ozone"]:
    if ((x < (mu - 3*sigma)) or (x > (mu + 3*sigma))):
        print(x)


# ### observations
# . it is observed that only two outliers are identified using std method
# . in box plot method more no of outliers are identified
# .this is because the assumption of normally is not satified in this column

# Quantile-Quantile plot for detection of ouliers

# In[66]:


import scipy.stats as stats
# create Q-Q plot
plt.figure(figsize=(8, 6))
stats.probplot(data1["Ozone"], dist="norm" , plot=plt)
plt.title("Q-Q plot for Outlier Detection", fontsize=14)
plt.xlabel("Theoretical Quantiles", fontsize=12)


# Other visualisation that could help in detection of outliers

# In[68]:


# create a figure for viol plot
sns.violinplot(data=data1["Ozone"], color='lightgreen')
plt.title("violin plot")


# In[ ]:





# In[70]:


plt.figure(figsize=(8, 6))
stats.probplot(data1["Solar"], dist="norm" , plot=plt)
plt.title("Q-Q plot for Outlier Detection", fontsize=14)
plt.xlabel("Theoretical Quantiles", fontsize=12)

