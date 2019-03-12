#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[21]:


from sklearn.datasets import load_digits
digits=load_digits()
dir(digits)


# In[37]:


X,y = digits.data[:-10], digits.target[:-10]


# In[57]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, 
                                                    random_state=1)


# In[58]:


from sklearn.svm import SVC
clf=SVC(C=1.0, gamma='auto', kernel='linear')


# In[59]:


clf.fit(X_train, y_train)


# In[67]:


clf.predict(digits.data[[8]])


# In[68]:


clf.score(X_test, y_test)


# In[69]:


plt.imshow(digits.images[8], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()


# In[ ]:




