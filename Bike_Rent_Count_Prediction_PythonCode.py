# %%
'''
# Project  :     " BIKE RENTAL COUNT PREDICTION"                            
'''

# %%
'''
## Problem Statement:
The objective of this Case is to Predication of bike rental count on daily based on the
environmental and seasonal settings.

The details of data attributes in the dataset are as follows -
instant: Record index

dteday: Date

season: Season (1:springer, 2:summer, 3:fall, 4:winter)

yr: Year (0: 2011, 1:2012)

mnth: Month (1 to 12)

hr: Hour (0 to 23)

holiday: weather day is holiday or not (extracted fromHoliday Schedule)

weekday: Day of the week

workingday: If day is neither weekend nor holiday is 1, otherwise is 0.

weathersit: (extracted fromFreemeteo)

1: Clear, Few clouds, Partly cloudy, Partly cloudy

2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist

3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered
clouds

4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog

temp: Normalized temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min),t_min=-8, t_max=+39 (only in hourly scale)

atemp: Normalized feeling temperature in Celsius. The values are derived via (t-t_min)/(t_maxt_min), t_min=-16, t_max=+50 (only in hourly scale)

hum: Normalized humidity. The values are divided to 100 (max)

windspeed: Normalized wind speed. The values are divided to 67 (max)

casual: count of casual users

registered: count of registered users

cnt: count of total rental bikes including both casual and registered

'''

# %%
# #!/usr/bin/env python
# #!pip install "library name"  # install relevent libraries with this command.
# # Importing required libraries. 

import os # input and output operations
import numpy as np # used for data analysis
import pandas as pd # for data manipulation and analysis
#import pandas_profiling # For Overview of data-summary statistics with plots
import pickle # Dumping your model in pkl format

# For data visualizations 
import matplotlib.pyplot as plt # used for data visualizations
import seaborn as sns # used for data visualizations
#from ggplot import * # used for data visualizations

# For rendering the plots in jupyter notebook

from random import randrange, uniform
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import math
from sklearn.linear_model import Lasso,Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import statsmodels.api as sm
from sklearn.model_selection import cross_val_score
from ggplot import *

# %%
# Set working directory

os.chdir("C:/Users/Sanjeev/Desktop/Edwisor/Bike_rent_Prediction_2nd_Project")
print(os.getcwd())

# %%
'''
# Understanding the data :
'''

# %%
bike_data =  pd.read_csv("day.csv")

# let's preview the training data
bike_data.head()

# %%
print("Shape of data is: ",bike_data.shape) #checking the number of rows and columns in training data


# %%
'''
# Overview of the training data  :
'''

# %%
'''
# Exploratory Data Analysis
'''

# %%
# Let's Check for data types of train data:
bike_data.info()


# %%
# Let's understand basic statistics of each (numeric & non-numeric) variables in train data
bike_data.describe()

# %%
print(bike_data.shape)
print(bike_data.columns)
print(bike_data.nunique())

# %%
# Check variables names of dataset,observed that shortcut names are used viz., hum for humidity
# yr for year ,mnth for month,cnt for count
# Now Rename them for ease of understanding it better:

print("Before renaming variables: ",bike_data.columns)

bike_data = bike_data.rename(columns = {'yr':'year','mnth':'month','weathersit':'weather',
                                        'temp':'temperature','hum':'humidity','cnt':'count'})


# %%
print("After renaming variables: ",bike_data.columns)

# %%
bike_data.head()

# %%
# In this dataset cnt is our target variable and it is continous variable 
bike_data['count'] .dtype

# %%
'''
# Missing Value anlysis
'''

# %%
# checking for missing values in dataset
bike_data.isnull().sum()

# %%
# Remove these variables 
# instant variable, as it is index in dataset
# date variable as we have to predict count on seasonal basis not date basis
# casual and registered variable as count is sum of these two variables
# cnt = casual + registered

bike_data = bike_data.drop(['casual','registered','instant','dteday'],axis=1)

# Lets check dimensions of data after removing unnecessary variables
bike_data.shape

# %%
# Make list of categorical and numerical variables of dataframe 

# Continous Variables 
cnames= ['temperature', 'atemp', 'humidity', 'windspeed', 'count']

# Categorical variables-
cat_cnames=['season', 'year', 'month', 'holiday', 'weekday', 'workingday','weather']


# %%
# looking at five point summary for our numerical variables
bike_data[cnames].describe()


# %%
# unique values of categories variables
bike_data[cat_cnames].nunique()

# %%
'''
# Outlier Analysis:
'''

# %%
# Lets save copy of dataset before preprocessing
df = bike_data.copy()
Bike_Rent = df.copy() 


##Plot boxplot to visulazie outliers-
A =['red','blue']
for i in cnames:
    print(i)
    sns.boxplot(y=Bike_Rent[i], color='red' )
    plt.xlabel(i)
    plt.ylabel("values")
    plt.title("Boxplot of "+i)
    plt.show()
    
# From boxplot we can see outliers in humidity and outliers in windspeed

# %%
# Lets cap outliers and inliers with upper fence and lower fence values 
for i in cnames:
    print(i)
    # Quartiles and IQR
    q25,q75 = np.percentile(Bike_Rent[i],[25,75])
    IQR = q75-q25
    
    # Lower and upper limits 
    LL = q25 - (1.5 * IQR)
    UL = q75 + (1.5 * IQR)
    
    # Capping with ul for maxmimum values 
    # For inliers
    Bike_Rent.loc[Bike_Rent[i] < LL ,i] = LL 

   # For outliers
    Bike_Rent.loc[Bike_Rent[i] > UL ,i] = UL 
     

# %%
# Lets see our boxplots after removing outliers 

for i in cnames:
    print(i)
    sns.boxplot(y=Bike_Rent[i], color='green')
    plt.xlabel(i)
    plt.ylabel("values")
    plt.title("Boxplot of "+i)
    plt.show()

# %%
'''
# Data understanding using visualization:
'''

# %%
'''
### Multi-Variate Analysis,    Univariate Analysis and   Bi-Variate Analysis of Train dataset
'''

# %%
'''
## Univariate Analysis:
'''

# %%
'''
#### Univariate Analysis : Displays the statistic details or descriptive statistics of each variable
#### Histogram for (Numeric) Continuous variables to check  distribution of each variable 
'''

# %%
# Histogram for continuous variables to check  distribution of each variable 
for i in cnames:
    sns.set(rc={'figure.figsize':(15,5)})
    sns.distplot(Bike_Rent[i], kde=True,  color='purple',bins ='auto')
    #plt.title(cnamestrain[i], fontsize=14)
   # plt.xlabel(cnamestrain[i], fontsize=14)
    plt.ylabel('Variable Frequency', fontsize=14)
    plt.show()

# %%
'''
## BI-VARIATE ANALYSIS :
'''

# %%
'''
### Bi-variate Analysis using Catplots:
'''

# %%
# Lets check impact of continous variables on target variable
for i in cat_cnames:
    sns.catplot(x=i,y="count",data=Bike_Rent,jitter='0.15',aspect=2.5)
    fname = str(i)+'.pdf'
    plt.savefig(fname) 

# %%
'''
*From Fisrt plot we can see that season 2,3 and 4 have more bike count as comapre to season 1. the daily bike count for these season was between 4000 to 8000.
*From year plot we can see that bike count is increased in 2012 as compared to 2011.
*From month plot we can see the bike count maximum between 4 to 10 month.
*From holiday the bike count is maximum as comapre to non holiday.
*Bike count is maximum for day 0,5 and 6 as per weekday varaible.
*FOr weather 1 the count of bike is maximum, after that for weather 2.
'''

# %%
'''
### Bi-variate Analysis using Barplots:
'''

# %%
## take a copy of data
df=bike_data
df.head()

# %%
df['actual_season'] = df['season'].replace({1:"Spring",2:"Summer",3:"Fall",4:"Winter"})
df['actual_holiday'] = df['holiday'].replace({0:"Working day",1:"Holiday"})
df['act_weather_condition'] = df['weather'].replace({1:"Clear",2:"Cloudy/Mist",3:"Light Rain/snow/Scattered clouds",4:"Heavy Rain/Snow/Fog"})
df['actual_weekday'] = df['weekday'].replace({0:"Monday",1:"Tuesday", 2:"Wednesday",3:"Thursday",4:"Friday",5:"Saturday",6:"Sunday"})
df['actual_year'] = df['year'].replace({0:"2011",1:"2012"})
df['actual_month'] = df['month'].replace({1:"Jan",2:'Feb',3:'March',4:'April',5:'May',6:'June',7:'July',8:'Aug',9:'Sept',10:'Oct',11:'Nov',12:'Dec'})

# %%
# Let us check impact of categorical variables on count
# For Catagorical Variables
fig, axarr = plt.subplots(4, 2, figsize=(15, 15))
plt.subplots_adjust(hspace=.3)

sns.barplot(x='actual_season', y='count',data=df,ax=axarr[0][0])
sns.barplot(x='actual_year',    y='count',data=df,ax=axarr[0][1])
sns.barplot(x='act_weather_condition',  y='count',data=df,ax=axarr[1][0])
sns.barplot(x='actual_holiday',y='count',data=df,ax=axarr[1][1])
sns.barplot(x='actual_weekday',     y='count',data=df,ax=axarr[2][0])
sns.barplot(x='workingday',  y='count',data=df,ax=axarr[2][1])

sns.barplot(x='actual_month',  y='count',data=df,ax=axarr[3][0])

# %%
df_Cnames=['actual_season','actual_holiday','act_weather_condition','actual_weekday','actual_year','actual_month']


# %%
# Let us check impact of categorical variables on count 
for i in df_Cnames:
    print ('frequency of count vs',i)
    print(df.groupby([i])['count'].sum(),'\n')

# %%
# Lets check impact of continous variables on target variable

f, ax = plt.subplots(figsize=(10, 10))
sns.scatterplot(x="temperature", y="count",
                hue="humidity", size="count",
                palette="rainbow",sizes=(1, 100), linewidth=0,
                data=Bike_Rent,ax=ax)
plt.title("Varation in bike rental count with respect to Normalized temperature humidity")
plt.ylabel("Bike rental count")
plt.xlabel("Normalized temperature")
plt.savefig('bike_temp&humidity_plot.pdf')

#*From the plot we can see that count is maximum when temprature 0.4 to 0.7 and humidity below 0.75.

# %%


# %%
f, ax = plt.subplots(figsize=(10, 10))
sns.scatterplot(x="temperature", y="count",
                hue="windspeed", size="humidity",
                palette="coolwarm",sizes=(1, 100), linewidth=0,
                data=Bike_Rent,ax=ax)
plt.title("Varation in bike rental count with respect to Normalized temperature windspeed")
plt.ylabel("Bike rental count")
plt.xlabel("Normalized temperature")
plt.savefig('bike_temp&windspeed_plot.pdf')

#*From the above plot we can see bike count is maximum between temp 0.5 to 0.7, windspped below 0.15 and humidity less than 0.75

# %%
f, ax = plt.subplots(figsize=(10, 10))
sns.scatterplot(x="temperature", y="count",
                hue="season", size="count",style= "weather",
                palette="winter",sizes=(1, 100), linewidth=0,
                data=Bike_Rent,ax=ax)
plt.title("Varation in bike rental count with respect to Normalized temperature and season")
plt.ylabel("Bike rental count")
plt.xlabel("Normalized temperature")
plt.savefig('bike_temp&season_plot.pdf')

#*From figure it is clear that maximum bike count is for season 2 and 3, when the temp between 0.5 to 0.7, and weather was 1 and 2

# %%
'''
### Bi-variate Analysis using Scatter plots:
'''

# %%
# Bi-variate Analysis: 
# Here one variable is independent while other one is dependent.So, here count is dependent variable and rest all variables are independent variables.
# let'scheck scatter plot for the variables with count how are they co-rrelated.

for i in cnames:
    sns.set(rc={'figure.figsize':(15,5)})
    ax = sns.scatterplot(x=Bike_Rent[i], y="count", data=Bike_Rent)
    #plt.title(cnames1[i], fontsize=14)
    #plt.xlabel(cnames1[i], fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.show()

# %%
'''
# Feature Selection  :
'''

# %%
Bike_Rent.shape

# %%
Bike_Rent.dtypes

# %%
print(Bike_Rent.columns)

# %%
# Correlation analysis
# Using corrplot library we do correlation analysis for numeric variables
# Lets recall numeric variabls and derive correlation matrix and plot

# Continous Variables 
cnames= ['temperature', 'atemp', 'humidity', 'windspeed', 'count']

# Correlation matrix 
# Extract only numeric variables in dataframe for correlation
df_corr= Bike_Rent.loc[:,cnames]

# Generate correlation matrix
corr_matrix = df_corr.corr()
(print(corr_matrix))


# %%
# Set the width and hieght of the plot
f, ax = plt.subplots(figsize=(7, 5))

#Plot using seaborn library
sns.heatmap(corr_matrix, mask=np.zeros_like(corr_matrix, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax,annot=True)

plt.title("Correlation Plot of Continous Variables")

# From correlation analysis temp and atemp variables are highly correlated 
# so delete atemp variable 

# %%
# Categorical variables-
cat_cnames=['season', 'year', 'month', 'holiday', 'weekday', 'workingday','weather']

# %%
# Lets find significant categorical variables usig ANOVA test 
# Anova analysis for categorical variable with target numeric variable

import statsmodels.api as sm
from statsmodels.formula.api import ols

for i in cat_cnames:
    mod = ols('count' + '~' + i, data = Bike_Rent).fit()
    aov_table = sm.stats.anova_lm(mod, typ = 2)
    print(aov_table)

# %%
'''
From the anova result, we can observe working day,weekday and holiday has p value > 0.05, so delete this variable not consider in model.
'''

# %%
'''
# Dimension reduction
'''

# %%
Bike_Rent = Bike_Rent.drop(['atemp','holiday','weekday','workingday'], axis=1)

# %%
# Lets check dimensions after dimension reduction 
Bike_Rent.shape

# %%
# Lets check column names after dimension reduction 
Bike_Rent.columns

# %%
Bike_Rent.head()

# %%
# Lets define/update  continous and categorical variables after dimension reduction

# Continuous variable
cnames = ['temperature','humidity', 'windspeed', 'count']

# Categorical variables
cat_cnames = ['season', 'year', 'month','weather']

# %%
'''
# Feature Scaling
'''

# %%
# Since as it is mentioned in data dictionary the values of 
# temp,humidity,windspeed variables are already normalized values 
# So no need to go for feature scaling instead we will visualize the variables 
# to see normality

# %%
'''
## Normality Check
'''

# %%
for i in cnames:
    print(i)
    sm.qqplot(Bike_Rent[i])
    plt.title("Normal qq plot of " +i)
    plt.show()

# %%
for i in cnames:
    print(i)
    sns.distplot(Bike_Rent[i],bins='auto',color='blue')
    plt.title("Distribution of Variable"+i)
    plt.ylabel("Density")
    plt.show()

# %%
Bike_Rent.loc[:,'count'] = Bike_Rent.loc[:,'count'].round()
Bike_Rent.describe()

# from distribution plot,normal qq plot  and summary  it is clear that data is already normalized.

# %%
Bike_Rent.dtypes

# %%
'''
# MODEL DEVELOPMENT
'''

# %%
# Load Required libraries for model development 

# For Machine learning 

from sklearn.model_selection import train_test_split #used to split dataset into train and test

from sklearn.metrics import mean_squared_error # used to calculate MSE

from sklearn.metrics import r2_score # used to calculate r square

from sklearn.linear_model import LinearRegression # For linear regression

from sklearn.tree import DecisionTreeRegressor # For Decision Tree

from sklearn.ensemble import RandomForestRegressor # For RandomForest

from sklearn import metrics

# %%
# Lets convert all categorical variables ito dummy variables 
# As we cant pass categorical variables directly in to regression problems
# Lets save our preprocessed data into df data set 

df1 = Bike_Rent
Bike_Rent = df1

# Lets call Categorical varaibles after feature selection using ANOVA 
cat_cnames = ['season', 'year', 'month','weather']

#  Create categorical variables to dummy variables-
Bike_Rent = pd.get_dummies(Bike_Rent,columns=cat_cnames)
Bike_Rent.head()

# %%
# To avoid dummy variable trap I am hoing to remove 1 dummy variable form each type of categorical variable
Dummy_Drop = ['season_4', 'year_1', 'month_12','weather_3']
Bike_Rent.drop(Dummy_Drop,axis=1,inplace=True)

# %%
# Before developing the model lets check the dimensions of data 
Bike_Rent.shape

# %%
Bike_Rent.head()

# %%
Bike_Rent.columns

# %%
# Lets Divide the data into train and test set 

# Split data for predictor and target seperatly
X= Bike_Rent.drop(['count'],axis=1)
y= Bike_Rent['count']

# %%
Bike_Rent.head()

# %%
# Now Split the data into train and test using train_test_split function
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=101)

# %%
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# %%
# Function for Error metrics to calculate the performance of model
def MAPE(y_true,y_prediction):
    mape= np.mean(np.abs(y_true-y_prediction)/y_true)*100
    return mape

# %%
## Function to print the characteristics of model
def Model_detail(MAPE_test,r2_test,RMSE_test):
    print("Mean Absolute Precentage Error for test data="+str(MAPE_test))
    print("R^2_score for test data="+str(r2_test))
    print("RMSE for test data="+str(RMSE_test))
    print("Accuracy :="+str(100-MAPE_test))
    return None

# %%
# Function for Error metrics to display the performance of model
def Error_Metrics(Model_name,MAPE_test,r2_test,RMSE_test):
    Error_Metrics = {'Model Name': [Model_name],'Accuracy':[100-MAPE_test],'MAPE_Test':[MAPE_test],
      'R-squared_Test':[r2_test],'RMSE_test':[RMSE_test]}
    Results = pd.DataFrame(Error_Metrics)
    return Results

# %%
# Before building multiple linear regression model lets check the 
# vif for multicolinearity
# continous variables after feature selection using correlation analysis 

# Import VIF function from statmodels Library
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Get variables for which to compute VIF and add intercept term:

X = Bike_Rent[["temperature","humidity","windspeed"]].dropna() #subset the dataframe
X ['Intercept'] = 1

# Compute and view VIF:

vif = pd.DataFrame()           # Create an empty dataframe
vif["Variables"] = X.columns   # Add "Variables" column to empty dataframe
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# View results using print
print(vif)

# %%
'''
None of the variables from the 3 input variables has collinearity problem.
'''

# %%
'''
# Linear Regression Model  :
'''

# %%
# Import libraries
import statsmodels.api as sm

# Linear Regression model for regression 
LR_model= sm.OLS(y_train,X_train).fit()
print(LR_model.summary())

# %%
# Model prediction on test data
LR_test= LR_model.predict(X_test)

# Model performance on test data
MAPE_test= MAPE(y_test,LR_test)

# r2 value for test data-
r2_test=r2_score(y_test,LR_test)

# RMSE value for test data
RMSE_test = np.sqrt(metrics.mean_squared_error(y_test,LR_test))

# Print the characteristics of model
Model_detail(MAPE_test,r2_test,RMSE_test)

# %%
LR_Results = Error_Metrics('Linear Regression',MAPE_test,r2_test,RMSE_test)


# %%
LR_Results

# %%
'''
Lets build some more models using different ml algorithms for more accuracy 
and less prediction error
'''

# %%
'''
# Desicision Tree
'''

# %%
# Lets Build decision tree model on train data
# Import libraries
from sklearn.tree import DecisionTreeRegressor

# Decision tree for regression
DT_model= DecisionTreeRegressor().fit(X_train,y_train)


# Model prediction on test data
DT_test= DT_model.predict(X_test)


# Model performance on test data
MAPE_test= MAPE(y_test,DT_test)


# r2 value for test data
r2_test=r2_score(y_test,DT_test)


# RMSE value for test data
RMSE_test = np.sqrt(metrics.mean_squared_error(y_test,DT_test))

# Print the characteristics of model
Model_detail(MAPE_test,r2_test,RMSE_test)

# %%
DT_Results = Error_Metrics('Desicision Tree',MAPE_test,r2_test,RMSE_test)

# %%
DT_Results

# %%
from sklearn import tree

# %%
'''
# Random Search CV In Decision Tree 
'''

# %%
# Import libraries 
from sklearn.model_selection import RandomizedSearchCV

RandomDecisionTree = DecisionTreeRegressor(random_state = 0)
depth = list(range(1,20,2))
random_search = {'max_depth': depth}

# Lets build a model using above parameters on train data 
RDT_model= RandomizedSearchCV(RandomDecisionTree,param_distributions= random_search,n_iter=5,cv=5)
RDT_model= RDT_model.fit(X_train,y_train)


# %%
# Lets look into best fit parameters
best_parameters = RDT_model.best_params_
print(best_parameters)

# %%
# Again rebuild decision tree model using randomsearch best fit parameter ie
# with maximum depth = 5
RDT_best_model = RDT_model.best_estimator_

# %%
# Prediction on train data 
RDT_train = RDT_best_model.predict(X_train)

# Prediction on test data 
RDT_test = RDT_best_model.predict(X_test)

# %%
# Lets check Model performance on train data using error metrics of regression like mape,rsquare value

# MAPE for test data 
MAPE_test= MAPE(y_test,RDT_test)


# Rsquare for test data
r2_test=r2_score(y_test,RDT_test)


# RMSE value for test data
RMSE_test = np.sqrt(metrics.mean_squared_error(y_test,RDT_test))


# Lets print the results 
print("Best Parameter="+str(best_parameters))
print("Best Model="+str(RDT_best_model))

# Print the characteristics of model
Model_detail(MAPE_test,r2_test,RMSE_test)

# %%
RDT_Results = Error_Metrics('Random Search CV In Decision Tree',MAPE_test,r2_test,RMSE_test)

# %%
RDT_Results

# %%
'''
# Grid Search CV in Decision Tree
'''

# %%
# Import libraries
from sklearn.model_selection import GridSearchCV

GridDecisionTree= DecisionTreeRegressor(random_state=0)
depth= list(range(1,20,2))
grid_search= {'max_depth':depth}

# Lets build a model using above parameters on train data
GDT_model= GridSearchCV(GridDecisionTree,param_grid=grid_search,cv=5)
GDT_model= GDT_model.fit(X_train,y_train)

# %%
# Lets look into best fit parameters from gridsearch cv DT
best_parameters = GDT_model.best_params_
print(best_parameters)

# %%
# Again rebuild decision tree model using gridsearch best fit parameter ie
# with maximum depth = 5
GDT_best_model = GDT_model.best_estimator_

# %%

# Prediction on train data  test data-
GDT_test = GDT_best_model.predict(X_test)

# %%
# Lets check Model performance on train data using error metrics of regression like mape,rsquare value

# MAPE for test data 
MAPE_test= MAPE(y_test,GDT_test)


# Rsquare for train data
r2_test=r2_score(y_test,GDT_test)


# RMSE value for test data
RMSE_test = np.sqrt(metrics.mean_squared_error(y_test,GDT_test))


print("Best Parameter="+str(best_parameters))
print("Best Model="+str(GDT_best_model))

# Print the characteristics of model
Model_detail(MAPE_test,r2_test,RMSE_test)

# %%
GDT_Results = Error_Metrics('Decision Tree Grid Search CV',MAPE_test,r2_test,RMSE_test)

# %%
GDT_Results

# %%
'''
# Random Forest 
'''

# %%
# Import libraris
from sklearn.ensemble import RandomForestRegressor

# Random Forest for regression
RF_model= RandomForestRegressor(n_estimators=100).fit(X_train,y_train)


# Prediction on test data
RF_test= RF_model.predict(X_test)


# MAPE For test data
MAPE_test= MAPE(y_test,RF_test)


# Rsquare  For test data
r2_test=r2_score(y_test,RF_test)


# RMSE value for test data
RMSE_test = np.sqrt(metrics.mean_squared_error(y_test,RF_test))

# Print the characteristics of model
Model_detail(MAPE_test,r2_test,RMSE_test)

# %%
# Lets print results of Randomforest random search
RF_Results = Error_Metrics('Random Forest',MAPE_test,r2_test,RMSE_test)

# %%
RF_Results

# %%
'''
# Random Search CV in Random Forest 
'''

# %%
# Import libraries
from sklearn.model_selection import RandomizedSearchCV

RandomRandomForest = RandomForestRegressor(random_state = 0)
n_estimator = list(range(1,100,2))
depth = list(range(1,20,2))
random_search = {'n_estimators':n_estimator, 'max_depth': depth}

# Lets build a model using above parameters on train data
RRF_model= RandomizedSearchCV(RandomRandomForest,param_distributions= random_search,n_iter=5,cv=5)
RRF_model= RRF_model.fit(X_train,y_train)

# %%
# Best parameters for model
best_parameters = RRF_model.best_params_
print(best_parameters)

# %%
# Again rebuild random forest  model using gridsearch best fit parameter ie {'n_estimators': 43, 'max_depth': 7}
RRF_best_model = RRF_model.best_estimator_

# %%
# Prediction on test data
RRF_test = RRF_best_model.predict(X_test)

# %%
# Lets check Model performance on train data using error metrics of regression like mape,rsquare value

# MAPE for test data
MAPE_test= MAPE(y_test,RRF_test)


# Rsquare for test data
r2_test=r2_score(y_test,RRF_test)


# RMSE value for test data
RMSE_test = np.sqrt(metrics.mean_squared_error(y_test,RRF_test))


print("Best Parameter="+str(best_parameters))
print("Best Model="+str(RRF_best_model))

# Print the characteristics of model
Model_detail(MAPE_test,r2_test,RMSE_test)

# %%
# Lets print results of Randomforest random search
RRF_results = Error_Metrics('Random Forest Random Search CV',MAPE_test,r2_test,RMSE_test)

# %%
RRF_results

# %%
'''
# Grid search CV in Random Forest 
'''

# %%
# Import libraries
from sklearn.model_selection import GridSearchCV

GridRandomForest= RandomForestRegressor(random_state=0)
n_estimator = list(range(1,20,2))
depth= list(range(1,20,2))
grid_search= {'n_estimators':n_estimator, 'max_depth': depth}

# %%
# Lets build a model using above parameters on train data using random forest grid search cv 
GRF_model= GridSearchCV(GridRandomForest,param_grid=grid_search,cv=5)
GRF_model= GRF_model.fit(X_train,y_train)

# %%
# Best fit parameters for model
best_parameters_GRF = GRF_model.best_params_
print(best_parameters_GRF)

# %%
# Again rebuild random forest model using gridsearch best fit parameter {'max_depth': 7, 'n_estimators': 11}
GRF_best_model = GRF_model.best_estimator_

# %%
# Prediction on test data
GRF_test = GRF_best_model.predict(X_test)

# %%
# Lets check Model performance on train data using error metrics of regression like mape,rsquare value

# MAPE for test data
MAPE_test= MAPE(y_test,GRF_test)

# Rsquare for test data
r2_test=r2_score(y_test,GRF_test)


# RMSE value for test data
RMSE_test = np.sqrt(metrics.mean_squared_error(y_test,GRF_test))

print("Best Parameter="+str(best_parameters))
print("Best Model="+str(GRF_best_model))

# Print the characteristics of model
Model_detail(MAPE_test,r2_test,RMSE_test)

# %%
# Lets print results of Randomforest grid search cv

GRF_results = Error_Metrics('Random Forest Grid Search CV',MAPE_test,r2_test,RMSE_test)

# %%
GRF_results

# %%
'''
# Gradient Boosting 
'''

# %%
# Import libraries
from sklearn.ensemble import GradientBoostingRegressor

# Lets build a Gradient Boosting model for regression problem
GB_model = GradientBoostingRegressor().fit(X_train, y_train)


# Model prediction on test data
GB_test= GB_model.predict(X_test)


# Model performance on test data
MAPE_test= MAPE(y_test,GB_test)


# Rsquare value for test data
r2_test=r2_score(y_test,GB_test)


# RMSE value for test data
RMSE_test = np.sqrt(metrics.mean_squared_error(y_test,GB_test))


# Print the characteristics of model
Model_detail(MAPE_test,r2_test,RMSE_test)

# %%
# Lets print the result 
GB_results = Error_Metrics('Gradient Boosting',MAPE_test,r2_test,RMSE_test)

# %%
GB_results 

# %%
'''
# Random Search CV in Gradient Boosting 
'''

# %%
# Import libraries
from sklearn.model_selection import RandomizedSearchCV

RandomGradientBoosting = GradientBoostingRegressor(random_state = 0)
n_estimator = list(range(1,100,2))
depth = list(range(1,20,2))
random_search = {'n_estimators':n_estimator, 'max_depth': depth}

# %%
# Lets build a model using above parameters on train data
RGB_model= RandomizedSearchCV(RandomGradientBoosting,param_distributions= random_search,n_iter=5,cv=5)
RGB_model= RGB_model.fit(X_train,y_train)

# %%
# Best parameters for model
best_parameters = RGB_model.best_params_
print(best_parameters)

# %%
# Again rebuild random forest model using gridsearch best fit parameter {'n_estimators': 81, 'max_depth': 5}
RGB_best_model = RGB_model.best_estimator_

# %%
# Prediction on test data
RGB_test = RGB_best_model.predict(X_test)

# %%
# Lets check Model performance on train data using error metrics of regression like mape,rsquare value


# MAPE for test data
MAPE_test= MAPE(y_test,RGB_test)


# Rsquare for test data
r2_test=r2_score(y_test,RGB_test)


# RMSE value for test data
RMSE_test = np.sqrt(metrics.mean_squared_error(y_test,LR_test))

print("Best Parameter="+str(best_parameters))
print("Best Model="+str(RGB_best_model))

# Print the characteristics of model
Model_detail(MAPE_test,r2_test,RMSE_test)

# %%
# Lets print the result 
RGB_results = Error_Metrics('Gradient Boosting Random Search CV',MAPE_test,r2_test,RMSE_test)

# %%
RGB_results

# %%
'''
# Grid Search CV in Gradient Boosting
'''

# %%
# Import libraries
from sklearn.model_selection import GridSearchCV

GridGradientBoosting= GradientBoostingRegressor(random_state=0)
n_estimator = list(range(1,20,2))
depth= list(range(1,20,2))
grid_search= {'n_estimators':n_estimator, 'max_depth': depth}

# %%
# Lets build a model using above parameters on train data(Grind Random Forest)
GGB_model= GridSearchCV(GridGradientBoosting,param_grid=grid_search,cv=5)
GGB_model= GGB_model.fit(X_train,y_train)

# %%
# Best parameters for model
best_parameters = GGB_model.best_params_
print(best_parameters)

# %%
# Again rebuild random forest model using gridsearch best fit parameter {'max_depth': 5, 'n_estimators': 19 }
GGB_best_model = GGB_model.best_estimator_

# %%
# Prediction on test data
GGB_test = GGB_best_model.predict(X_test)

# %%
# Lets check Model performance on both test and train using error metrics of regression like mape,rsquare value

# MAPE for test data
MAPE_test= MAPE(y_test,GGB_test)


# Rsquare value for test data
r2_test=r2_score(y_test,GGB_test)


# RMSE value for test data
RMSE_test = np.sqrt(metrics.mean_squared_error(y_test,GGB_test))

print("Best Parameter="+str(best_parameters))
print("Best Model="+str(GGB_best_model))

# Print the characteristics of model
Model_detail(MAPE_test,r2_test,RMSE_test)

# %%
# Lets print the results
GGB_Results = Error_Metrics('Gradient Boosting Grid Search CV',MAPE_test,r2_test,RMSE_test)

# %%
GGB_Results 

# %%
Final_Results = pd.concat([LR_Results,DT_Results,RDT_Results,GDT_Results,RF_Results,RRF_results,GRF_results,GB_results,RGB_results,GGB_Results], ignore_index=True, sort =False)

# %%
Final_Results

# %%
# From above results Gradient Boosting model have optimum values and this algorithm is good for our data 
# Lets save the output of finalized model (GB)

input = y_test.reset_index()
pred = pd.DataFrame(GB_test,columns = ['Predicted_Bike_Rental_Count'])
Final_output = pred.join(input)
Final_output = Final_output.rename(columns = {"count": "Actual_Bike_Rental_Count"})
Final_output.loc[:,'Predicted_Bike_Rental_Count'] = Final_output.loc[:,'Predicted_Bike_Rental_Count'].round()
Final_output = Final_output.drop('index',axis=1)


# %%
GB_model

# %%
#import pickle
pickle.dump(GB_model, open('model.pkl','wb'))

# %%
Final_output = Final_output.sort_values(['Actual_Bike_Rental_Count'], ascending=[True])
Final_output

# %%
Final_output.to_csv("C:/Users/Sanjeev/Desktop/Edwisor/Bike_rent_Prediction_2nd_Project/Bike_Rental_Count_GB_results_py.csv")


# %%
Final_Results.to_csv("C:/Users/Sanjeev/Desktop/Edwisor/Bike_rent_Prediction_2nd_Project/Bike_Rental_Count_Model_Summary_py.csv")

# %%
'''
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ END ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
'''

# %%
