#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np


# In[2]:


cars = pd.read_csv("Cars.csv")
cars.head()


# In[3]:


caes= pd.DataFrame(cars, columns=["HP","VOL","SP","WT","MP6"])
cars.head()


# In[4]:


cars.info()


# In[9]:


cars.isna().sum()


# In[ ]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# 

# In[ ]:





# In[ ]:





# In[ ]:


### Observation from boxplot and histograms
. There are some extreme values (outliers) observed in towards the right tail of sp and Hp distribution
. in VOL and WT columns, a few outliers are observed in both tails of there distribution
. The extreme values of cars data may have come from the special design nature of cars
. As this is multi-dimensional data, the outliers with respect to spatial dimensions may have to be comsidered


# In[11]:


cars[cars.duplicated()]


# ### pair plots and Correlation Coefficients

# In[15]:


sns.set_style(style='darkgrid')
sns.pairplot(cars)


# In[ ]:




