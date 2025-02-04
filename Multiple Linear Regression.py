#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np


# In[7]:


cars = pd.read_csv("Cars.csv")
cars.head()


# In[9]:


caes= pd.DataFrame(cars, columns=["HP","VOL","SP","WT","MP6"])
cars.head()

