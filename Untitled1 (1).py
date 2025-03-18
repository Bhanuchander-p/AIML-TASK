#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler


# In[2]:


df = pd.read_csv('diabetes.csv')
df


# In[3]:


x = df.drop('class', axis=1)
y = df['class']


# In[4]:


scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
x_scaled


# In[19]:


x_train, x_test, y_train, y_test = train_test_split(x_scaled,y, test_size = 0.2, random_state = 42)

                                                    


# In[20]:


gbc = GradientBoostingClassifier(random_state=42)


# In[21]:


kflod = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# In[22]:


param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3,4,5],
    'subsample': [0.8, 1.0]
}


# In[23]:


grid_search = GridSearchCV(estimator=gbc,
                           param_grid=param_grid,
                           cv=KFold,
                           scoring='recall',
                           n_jobs=-1,
                           verbose=1)


# In[18]:


grid_search.fit(x_train, y_train)
print("Best parameters:", grid_search.best_params_)
print("Best Cross-validated Recall:", grid_search.best_score_)


# In[ ]:




