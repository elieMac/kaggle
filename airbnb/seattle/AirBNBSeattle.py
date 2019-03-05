#------------------------------------------------------------------------------
#AirBNB notebook
#------------------------------------------------------------------------------

#%% [markdown]
# AIRBNB NOTEBOOK

#%% [markdown]
## Libraries
#%%
#Get the Basics
import math
import numpy as np
import pandas as pd
pd.set_option('display.max_columns',100)
import datetime
import pygments

#Visualization
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style as style
import folium as fo

#Scaling nicely
from sklearn.preprocessing import StandardScaler

#Models to be used
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV

#Select relevant features
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

#Select best model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
%matplotlib inline
#------------------------------------------------------------------------------

#%% [markdown]
## Data Engineering
#%%
calendar = pd.read_csv('calendar.csv')
listings = pd.read_csv('listings.csv')
reviews = pd.read_csv('reviews.csv')

#%%[markdown]
### Check if there is null data

#%%
print(calendar.shape)
calendar.isnull().sum()

#%%
print(listings.shape)
listings.isnull().sum()

#%%
listings_num = []
for column in listings.columns:
  if listings[column].dtype in ['int64','float64']:
      listings_num.append(column);

listings_numerical = listings[listings_num];
listings_numerical.describe()

# Scrape Id as a null std, check if any other data is useless/corrupted
null_std= []
for column in listings_numerical.columns:
    if 0 == listings_numerical[column].std():
        null_std.append(column);
    elif math.isnan(listings_numerical[column].std()):
        null_std.append(column);
    elif listings_numerical[column].min() == listings_numerical[column].max():
        null_std.append(column);

for element in null_std:
    listings_num.remove(element);

listings_numerical = listings[listings_num];
listings_numerical.describe()

#%%
listings_numerical.head(3)

#%%
listings_categorical = listings[listings.columns.difference(listings_num)]
listings_categorical.head(3)

#%%
listings_categorical = listings_categorical[listings_categorical.columns.difference(['country','country_code','city','smart_location','state','calendar_last_scraped','scrape_id'])];
listings_categorical.head(3)

#%% [markdown]
## Data Understanding
#%%
# Compute the correlation matrix
corr = listings_numerical.corr()

#%%
# Generate a mask for the upper triangle
#mask = np.zeros_like(corr, dtype=np.bool)
#mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(15, 15))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
hm = sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5});
hm.set_xticklabels(labels=hm.get_xticklabels(),rotation=75);

#%% [markdown]

##%%
print(reviews.shape)
reviews.isnull().sum()

#
##%%
#def get_month(date):
#    new_date = datetime.datetime.strptime(date, "%Y-%m-%d")
#    return new_date.strftime("%b")
#
#def get_day(date):
#    new_date = datetime.datetime.strptime(date, "%Y-%m-%d")
#    return new_date.strftime("%b")
#
#def get_year(date):
#    new_date = datetime.datetime.strptime(date, "%Y-%m-%d")
#    return new_date.year
#
#def bool_tonum(b):
#    if b == 't':
#        return 1
#    else:
#        return 0
#
##%%
##Check Availability
#availability = calendar[['date', 'available']]
#availability['month'] = availability['date'].apply(get_month)
#availability['available']= availability['available'].apply(bool_tonum)
#availability = availability[['month','available']]
#availability = availability.groupby('month')['available'].sum()
#
##%%
##availability.plot()
#
##%%
#data = listings[['latitude','longitude']]
#data.isnull().sum()
#data.shape
#
##%%
#init = [i for i in data.iloc[0].values]
#map_osm = fo.Map(location=init, zoom_start=12)
#for row in data.values :
#    fo.CircleMarker(location=[row[0], row[1]],radius=10,popup='test',color='crimson',fill=False).add_to(map_osm)
#


