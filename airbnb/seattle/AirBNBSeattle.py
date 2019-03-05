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
#------------------------------------------------------------------------------
#%%
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

#%% [markdown]
## Helper functions
#%%
def remove_dollar(data):
    data = data.apply(str)
    data = [float(x.strip('$').replace(',','')) for x in data]
    return data

def convert_boolstr(data):
    return data.map(dict(t=1, f=0))

#%% [markdown]
## Data Engineering
#%%
calendar = pd.read_csv('calendar.csv')
listings = pd.read_csv('listings.csv')
reviews = pd.read_csv('reviews.csv')

#%%[markdown]
### Check if there is null data

#%% [markdown]
### Calendar

#%% 
print(calendar.shape)
calendar.isnull().sum()
calendar.head()

#%%
# Formating calendar data to be useful
calendar.price = remove_dollar(calendar.price)
calendar.available = convert_boolstr(calendar.available)
calendar.head()

#%% [markdown]
### Listings
#%%
print(listings.shape)
listings.isnull().sum()
listings.head()

#%%
# Formating listings data to be useful
# Checking if data has variance
for column in listings.columns.values:
    print("-{}- unique values: {}".format(column,len(listings[column].unique())))

#%%
# Removing uselss data
for column in listings.columns.values:
    if len(listings[column].unique()) == 1:
        listings = listings.drop(column,axis=1)
        print("Removed ",column)

listings.shape
listings.describe()

#%%
# Converting bool and removing dollars
boolean_str =['host_has_profile_pic','host_identity_verified','is_location_exact','instant_bookable',
              'require_guest_profile_picture','require_guest_phone_verification']
prices = ['price','weekly_price','monthly_price','security_deposit','cleaning_fee','extra_people']
for column in boolean_str:
    listings[column] = convert_boolstr(listings[column])

for column in prices:
    listings[column] = remove_dollar(listings[column])

#%%
# Display
listings.head(10)

#%% [markdown]
### Reviews
#%%
print(reviews.shape)
reviews.isnull().sum()
reviews.head()

#%% [markdown]
## Data Understanding
#%%
# Compute the correlation matrix
numeric_fields = []
for column in listings.columns.values:
    if listings[column].dtype in ['int64','float64']:
        numeric_fields.append(column)

listings_numerical = listings[numeric_fields]
corr = listings_numerical.corr()

#%%
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(15, 15));
ax.tick_params(axis='x', colors='white');
ax.tick_params(axis='y', colors='white');

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
hm = sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,
                 square=True, linewidths=.5, cbar_kws={"shrink": .5});
hm.set_xticklabels(labels=hm.get_xticklabels(),rotation=75);
cb = hm.collections[0].colorbar
cb.ax.tick_params(axis='y', colors='white');

## Classification
#%% [markdown]
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


