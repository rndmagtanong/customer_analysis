import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

path = 'C:/Users/Eo/Desktop/Coding/customer_analysis/customer_personality.csv'

data = pd.read_csv(path)

'''
Preprocess the features:
Some of the features can be simplified (ex. Year_Birth to Current_Age)
Some values can be simplified (Marital_Status and Education)
Some can be added together to create new features (Kidhome + Teenhome or Products)
Renaming column names also works for clarity

Note: For time-related features, this dataset is from 2014. Treat today as 2014-12-31
'''

### data preprocessing

## age
from datetime import date
# dataset is from 2014, so treat as if today is 2014-12-31
today = date(2014, 12, 31)
year = today.year
data['age'] = year - data['Year_Birth']

## education
education = list(data['Education'].unique())
# ['Graduation', 'PhD', 'Master', 'Basic', '2n Cycle']
# Can split into Undergraduate and Postgraduate
data['Education'] = data['Education'].replace({'Graduation' : 'Postgraduate', 'PhD' : 'Postgraduate', 'Master' : 'Postgraduate', 'Basic' : 'Undergraduate', '2n Cycle' : 'Undergraduate'})
education = list(data['Education'].unique())
# ['Postgraduate', 'Undergraduate']

## marital status
marital_status = list(data['Marital_Status'].unique())
# ['Single', 'Together', 'Married', 'Divorced', 'Widow', 'Alone', 'Absurd', 'YOLO']
# Civil Statuses are Single, Married, Separated/Divorced, or Widowed
data['Marital_Status'] = data['Marital_Status'].replace({'Together' : 'Married', 'Alone' : 'Single', 'Absurd' : 'Single', 'YOLO' : 'Single'})

## children
data['Children'] = data['Kidhome'] + data['Teenhome']
# only goes up to 3 (thankfully)

## length of service in days
data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'], dayfirst = True, infer_datetime_format = True)
data['Days'] = pd.to_numeric(data['Dt_Customer'].dt.date.apply(lambda x: (today - x)).dt.days, downcast = 'integer')

## total expenditure
data['Expenditure'] = data['MntWines'] + data['MntFruits'] + data['MntMeatProducts'] + data['MntFishProducts'] + data['MntSweetProducts'] + data['MntGoldProds']

## renaming some columns
data = data.rename(columns = {'MntWines' : 'Wines', 'MntFruits' : 'Fruits', 'MntMeatProducts' : 'Meat', 'MntFishProducts' : 'Fish', 'MntSweetProducts' : 'Sweets', 'MntGoldProds' : 'Gold'})

## creating final dataframe to work with
final_data = data[['age', 'Education', 'Marital_Status', 'Income', 'Expenditure', 'Days', 'Children', 'Wines', 'Fruits', 'Meat', 'Fish', 'Sweets', 'Gold']]
final_data.columns = final_data.columns.str.lower()

# look for nan values
print(final_data.isna().sum().sum())
# 24 null values
final_data = final_data.dropna()

# write to csv
final_data.to_csv('customer_data.csv', index = False)

### data analysis

## correlation matrix
corrmat = final_data.corr()
f, ax = plt.subplots(figsize = (15, 15))
sns.heatmap(corrmat, vmax = 0.8, annot = True, square = True, xticklabels = 1, yticklabels = 1, annot_kws = {"fontsize" : 8})
ax.set_title('customer_data heatmap')

## box plots for categorical features
categorical = ['education', 'marital_status']
for feature in categorical:
    title = str(feature) + ' vs. expenditure'
    box_data = pd.concat([final_data['expenditure'], final_data[feature]], axis = 1)
    f, ax = plt.subplots(figsize = (8, 6))
    fig = sns.boxplot(x = feature, y = 'expenditure', data = box_data)
    fig.set_title(title)
    fig.axis(ymin = 0, ymax = 2600)


## clustering
'''
Main types of customers:
1. Old Customers, High Spending Nature (Stars)
2. New Customers, High Spending Nature (Potential)
3. Old Customers, Low Spending Nature (Leakage)
4. New Customers, Low Spending Nature (Attention)
'''

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.mixture import GaussianMixture

scaler = StandardScaler()
data_temp = final_data[['income', 'days', 'expenditure']]
X_std = scaler.fit_transform(data_temp)
X = normalize(X_std, norm = 'l2')

gmm = GaussianMixture(n_components = 4, covariance_type = 'spherical', max_iter = 2000, random_state = 1).fit(X)
labels = gmm.predict(X)
data_temp['cluster'] = labels
data_temp = data_temp.replace({0 : 'Attention', 1 : 'Stars', 2 : 'Potential', 3 : 'Leakage'})

summary = data_temp[['income', 'days', 'expenditure', 'cluster']]
summary.set_index("cluster", inplace = True)
summary = summary.groupby('cluster').describe().transpose()
print(summary)

## box plot
f, ax = plt.subplots(figsize = (8, 6))
fig = sns.boxplot(x = 'cluster', y = 'expenditure', data = data_temp)
fig.set_title('cluster vs. expenditure')
fig.axis(ymin = 0, ymax = 2600)

'''
Some insights:
1. Naturally, income is highly corrolated to expenditure
2. More days means more expenditure
3. Customers with children are much less likely to spend on wine
4. Wine spenders more often than not also buy meat
5. The biggest wine customers have an average income of 69586 dollars
6. The biggest wine customers spend an average of 1252 dollars
7. People use the service for roughly 720 days
8. People with degrees spend more
9. Marital status does not really affect expenditure
'''