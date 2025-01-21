#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[3]:


df = pd.read_csv("Universities.csv")
df


# In[5]:


np.mean(df["SAT"])


# In[7]:


np.median(df["SAT"])


# In[ ]:




