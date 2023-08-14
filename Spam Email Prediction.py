#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


# # Data collection and Pre Processing

# In[2]:


mail_data = pd.read_csv(r"C:\Users\Shaddu\Downloads\spam.csv", encoding='latin-1')


# In[3]:


mail_data.head()


# In[4]:


mail_data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis = 1, inplace = True)


# In[5]:


mail_data.head(10)


# In[6]:


mail_data.shape


# In[7]:


mail_data.rename(columns={'v1':'category','v2':'message'},inplace=True)


# In[8]:


mail_data.head()


# Label Encoding

# In[9]:


mail_data.loc[mail_data['category']=='spam','category',] = 0
mail_data.loc[mail_data['category']=='ham','category',] = 1


# In[10]:


mail_data.head()


# # Data Visualization

# In[11]:


mail_data['category'].value_counts().plot(kind='bar')


# In[37]:


category_counts = mail_data['category'].value_counts()
colors = ['yellow', 'red']
plt.pie(category_counts, labels=['ham', 'spam'], colors=colors, autopct='%0.2f%%')
plt.title('Email Category Distribution')
plt.show()


# In[32]:


X = mail_data['message']

Y = mail_data['category']


# In[14]:


print(X)


# In[15]:


print(Y)


# # Splitting data into train and test

# In[16]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)


# In[17]:


print(X.shape)
print(X_train.shape)
print(X_test.shape)


# # Feature Extraction

# In[18]:


#transform the text data to feature vectors that can be used as input to the logistic regression
feature_extraction = TfidfVectorizer(min_df = 1, stop_words = 'english', lowercase = True)

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

#convert y_train and y_test values as integers
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')


# In[19]:


print(X_train_features)


# # Training the Model

# Logistic regression

# In[20]:


model = LogisticRegression()


# In[21]:


model.fit(X_train_features, Y_train)


# # Evaluating the trained model

# In[22]:


#prediction on training data

prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)


# In[23]:


print('Accuracy on training data :', accuracy_on_training_data)


# In[24]:


#prediction on test data

prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)


# In[25]:


print('Accuracy on test data :', accuracy_on_test_data)


# # Building a predictive system

# In[26]:


input_mail = ["07732584351 - Rodger Burns - MSG = We tried to call you re your reply to our sms for a free nokia mobile + free camcorder. Please call now 08000930705 for delivery tomorrow"]

input_data_features = feature_extraction.transform(input_mail)

#making predictions

prediction = model.predict(input_data_features)
print(prediction)

if (prediction[0] == 1):
    print("Ham mail")

else:
     print("Spam mail")   

