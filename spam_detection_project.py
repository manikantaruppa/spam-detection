#!/usr/bin/env python
# coding: utf-8

# In[44]:


#import modules
import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# In[3]:


#importing dataframe
df=pd.read_csv('SMSSpamCollection','\t',names=['label','messages'])


# In[4]:


df.head()


# In[32]:


#cleaning and data preprocessing
ps=PorterStemmer()
wordnet=WordNetLemmatizer()
corpus=[]
for i in range(len(df)):
    review=re.sub('[^a-zA-Z]',' ',df['messages'][i])
    review=review.lower()
    review=review.split()
    review= [wordnet.lemmatize(word)for  word in review if not word in set(stopwords.words('english'))]
    review= ''.join(review)
    corpus.append(review)


# In[33]:


#tfidf vectorizer
tf=TfidfVectorizer()
X=tf.fit_transform(corpus).toarray()


# In[34]:


#bag of words
#cv=CountVectorizer()
#X=cv.fit_transform(corpus).toarray()


# In[19]:


Y=pd.get_dummies(df['label'])


# In[20]:


Y=Y.iloc[:,1].values


# In[42]:


#splitting the data into train and test
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.20,random_state=0)


# In[41]:


#model prediction on naive bayes
spam_detect_model=MultinomialNB().fit(x_train,y_train)
y_pred=spam_detect_model.predict(x_test)


# In[46]:


spam_detect=SVC().fit(x_train,y_train)
y_pre=spam_detect.predict(x_test)


# In[40]:


#confusion matrix
confusion=confusion_matrix(y_test,y_pred)


# In[39]:


#Accuracy of the model
accuracy=accuracy_score(y_test,y_pred)
accuracy


# In[47]:


#confusion matrix
confusion=confusion_matrix(y_test,y_pre)


# In[48]:


#Accuracy of the model
accuracy=accuracy_score(y_test,y_pre)
accuracy


# In[ ]:




