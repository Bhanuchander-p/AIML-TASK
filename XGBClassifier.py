#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install xgboost


# In[2]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# In[3]:


df = pd.read_csv('diabetes.csv')
df


# In[11]:


x = df.drop('class', axis=1)
y = df['class']


# In[8]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[10]:


scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
print(x_train_scaled)
print("------------------------------------------------")
print(x_test_scaled)


# In[14]:


xbg = XGBClassifier(use_label_encoder=False, eval_metric='logloss',random_state=42)


# In[19]:


param_grid = {
    'n_estimators': [100, 150, 200, 300],
    'learning_rate': [0.01, 0.1, 0.15],
    'max_depth': [2,3,4,5],
    'subsample': [0.8, 1.0],
    'colsample_bytree':[0.8,1.0],
}
    skf = stratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# In[18]:


grid_search.fit(x_train_scaled, y_train)

