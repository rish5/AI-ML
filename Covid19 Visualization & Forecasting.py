#!/usr/bin/env python
# coding: utf-8

# In[15]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

## iplot 
from plotly.offline import iplot, init_notebook_mode
## cufflinks - transforming data for iplot
import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected= True)


# In[2]:


covid_confirmed = pd.read_csv('https://raw.githubusercontent.com/ammishra08/COVID-19/master/covid_19_datasets/covid_19_globaldata/time_series_covid_19_confirmed.csv')
covid_deaths = pd.read_csv('https://raw.githubusercontent.com/ammishra08/COVID-19/master/covid_19_datasets/covid_19_globaldata/time_series_covid_19_deaths.csv')
covid_recovered = pd.read_csv('https://raw.githubusercontent.com/ammishra08/COVID-19/master/covid_19_datasets/covid_19_globaldata/time_series_covid_19_recovered.csv')


# In[3]:


covid_confirmed


# In[4]:


covid_confirmed.rename({'Province/State':'State', 'Country/Region':'Country'}, axis=1, inplace=True)
covid_deaths.rename({'Province/State':'State', 'Country/Region':'Country'}, axis=1, inplace=True)
covid_recovered.rename({'Province/State':'State', 'Country/Region':'Country'}, axis=1, inplace=True)


# In[5]:


covid_confirmed.head(15)


# In[6]:


covid19_country = covid_confirmed.drop(['State','Lat','Long'],axis=1)


# In[7]:


covid19_country.head()


# In[8]:


## Group by to merge country data 
covid19_country = covid19_country.groupby(['Country']).sum()


# In[9]:


covid19_country.head()


# In[10]:


## Total Number of cases
covid19_country.iloc[:, -1].sum()


# In[11]:


covid19_country.sort_values(by = covid19_country.columns[-1], ascending=False).head(10)


# In[12]:


covid19_country.sort_values(by = covid19_country.columns[-1], ascending=False).head(10).transpose().plot(figsize=(15,10))


# In[17]:


covid19_country.sort_values(by = covid19_country.columns[-1], ascending=False).head(10).transpose().iplot(title = "Time Series Covid19 Confirmed cases in Top 10 Countries")


# In[18]:


covid19_country.loc['India'].transpose().iplot()


# In[19]:


covid19_country.loc['India'].diff().iplot(title = "Daily Increase in Number of Cases Reported in India")


# ### World Map Visualization

# In[20]:


import folium


# In[21]:


covid_confirmed


# In[25]:


world_map = folium.Map (location= [10,0], zoom_start= 2, max_zoom=8, min_zoom=1, width='100%', tiles = 'CartoDB dark_matter')
for i in range(0, len(covid_confirmed)):
    folium.Circle(location= [covid_confirmed.iloc[i]['Lat'], covid_confirmed.iloc[i]['Long']],
                 radius = (int(np.log(covid_confirmed.iloc[i, -1]+ 1.00001)))*30000, 
                  tooltip = "<h4 style='text-align:center;font-weight: bold'>" + covid_confirmed.iloc[i]['Country'] + "</h4>" +
                  "<div style='text-align:center;font-weight: bold''>"+str(np.nan_to_num(covid_confirmed.iloc[i]['State']))+"</div>"+
                  "<li>Confimed "+ str(covid_confirmed.iloc[i, -1])+"</li>"+
                  "<li>Deaths "+ str(covid_deaths.iloc[i, -1])+"</li>",
                  color = 'red', fill = True).add_to(world_map)
world_map


# In[ ]:




