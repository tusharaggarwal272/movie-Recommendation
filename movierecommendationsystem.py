#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import warnings


# In[3]:


warnings.filterwarnings('ignore')


# In[4]:


column_name=["user_id","item_id","rating","timestamp"]
df=pd.read_csv("u.data",sep='\t',names=column_name)


# In[5]:


df.head()


# In[6]:


df.shape


# In[7]:


df['item_id'].nunique()


# In[8]:


df1=pd.read_csv("u.item",sep='\|',header=None)


# In[9]:


df1=df1[[0,1]]


# In[10]:



df1.columns=["item_id","title"]
df1.head()


# In[11]:


df2=pd.merge(df,df1,on="item_id")


# In[12]:


df2.head()


# In[13]:


import matplotlib.pyplot as plt 
import seaborn as sns
sns.set_style('white')


# In[14]:


df2.groupby('title').mean()['rating'].sort_values(ascending=False)


# In[15]:


df2.groupby('title').count()['rating'].sort_values(ascending=False)


# In[16]:


ratings=pd.DataFrame(df2.groupby('title').mean()['rating'])


# In[17]:


ratings


# In[18]:


ratings['number of ratings']=df2.groupby('title').count()['rating']


# In[19]:


ratings


# In[20]:


ratings.sort_values(by='number of ratings',ascending =False)


# In[21]:


plt.figure(figsize=(10,6))
plt.hist(ratings['number of ratings'],bins=70)
plt.show()


# In[22]:


plt.hist(ratings['rating'],bins=70)
plt.show()


# In[23]:


sns.jointplot(x='rating',y='number of ratings',data=ratings)
plt.show()


# In[24]:


df2.head()


# In[25]:


moviematrix=df2.pivot_table(index="user_id",columns="title",values="rating")


# In[26]:


moviematrix


# In[27]:


star_wars=moviematrix['Star Wars (1977)']


# In[49]:


similar_to_starwars=moviematrix.corrwith(star_wars)


# In[50]:


similar_to_starwars


# In[51]:


similar_to_starwars=pd.DataFrame(similar_to_starwars,columns=["corelation"])


# In[52]:


similar_to_starwars


# In[53]:


correlation_with_othermovies=similar_to_starwars.dropna()


# In[55]:


correlation_with_othermovies


# In[58]:


correlation_with_othermovies.sort_values('corelation',ascending=False).head(10)


# In[59]:


ratings


# In[66]:


table=correlation_with_othermovies.join(ratings['number of ratings'])
table


# In[68]:


table[table['number of ratings']>100].sort_values('corelation',ascending=False)


# In[74]:


def prediction(moviename):
    movieuserratings=moviematrix[moviename]
    similartomovie=moviematrix.corrwith(movieuserratings)
    corr_movie=pd.DataFrame(similartomovie,columns=['corelation'])
    corr_movie.dropna(inplace=True)
    corr_movie=corr_movie.join(ratings['number of ratings'])
    predictions=corr_movie[corr_movie['number of ratings']>100].sort_values('corelation',ascending=False)
    return predictions


# In[78]:


prediction("Titanic (1997)").head()


# In[ ]:




