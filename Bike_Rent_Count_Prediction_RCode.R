# Clean the environment:
rm(list=ls())

# Load the required libraries for analysis of data-
x = c('randomFroest','caret','dummies','ggplot2', 'corrgram', 'corrplot', 'randomForest','caret', 'car','PerformanceAnalytics','class', 'e1071', 'rpart', 'mlr','grid',
      'DMwR','usdm','dplyr','caTools','LiblineaR',"randomForest", "unbalanced","C50","dummies", 
      "Information", "MASS", "rpart", "gbm", "ROSE",'sampling', 'DataCombine', 'inTrees',"scales","psych","gplots")

#install.packages("x")
# Check whether Required libraries are loaded into environment:
lapply(x, require, character.only = TRUE)
rm(x)

# Set working directory:
setwd("C:/Users/Sanjeev/Desktop/Edwisor/Bike_rent_Prediction_2nd_Project")
getwd()

# Load the training data :
bike_data <- read.csv("day.csv", header = T,na.strings = c(""," ",NA))  # stringsAsFactors

# let's preview the training data
head(bike_data)

#Check the dimensions(no of rows and no of columns)
print(paste(" Shape of training data is :   No. of Rows =",nrow(bike_data)," And No. of Columns =", ncol(bike_data)," "))

##### Overview of the historical data  #
##    Exploratory Data Analysis    ###

# Understanding complete historical data of two years 2011 and 2012:
dim(bike_data)
class(bike_data)
names(bike_data)
str(bike_data)
head(bike_data)
summary(bike_data)

# Check variables names of dataset,observed that shortcut names are used viz., hum for humidity
# yr for year ,mnth for month,cnt for count
# Now Rename them for ease of understanding it better

print(paste(" Before Renaming of variables : ",names(bike_data)))

#Rename the variables-
names(bike_data)[4]  = "year"
names(bike_data)[5]  = "month"
names(bike_data)[9]  = "weather"
names(bike_data)[10] = "temperature"
names(bike_data)[12] = "humidity"
names(bike_data)[16] = "count"

print(paste(" After Renaming of variables : ",names(bike_data)))

head(bike_data)

##  Variable Identification  ##
# In this dataset cnt is our target variable and it is continous variable 
str(bike_data$count) 


###  Missing Value anlysis  ###

# Check missing values in dataset
sum(is.na(bike_data))   # there is no missing values present in this dataset

# Remove these variables 
# instant variable, as it is index in dataset
# date variable as we have to predict count on seasonal basis not date basis
# casual and registered variable as count is sum of these two variables
# cnt = casual + registered 

bike_data = subset(bike_data,select=-c(instant,dteday,casual,registered))

# Lets check dimensions of data after removing some variables
dim(bike_data)

# Make list of categorical and numerical variables of dataframe 
# Continous Variables 
cnames= c("temperature","atemp","humidity","windspeed","count")

# Categorical varibles-
cat_cnames= c("season","year","month","holiday","weekday","workingday","weather")

# looking at five point summary for our numerical variables
summary(bike_data)

sapply(bike_data[cat_cnames], function(x) length(unique(x)))

#### Outlier Analysis  ###
# Lets save copy of dataset before preprocessing
df = bike_data 
Bike_Rent = df 

# Lets use boxplot to detect the outliers 
# We use ggplot library to plot boxplot for each numeric variable 
col = rainbow(ncol(Bike_Rent))
for(i in 1:length(cnames))
{
  assign(paste0("gn",i),ggplot(aes_string(y=(cnames[i]),x = 'count'),
                               data=subset(Bike_Rent))+
           stat_boxplot(geom = "errorbar",width = 0.5) +
           geom_boxplot(outlier.color = "red",fill=col[i],
                        outlier.shape = 18,outlier.size = 1,notch = FALSE)+
           theme(legend.position = "bottom")+
           labs(y = cnames[i],x='count')+
           ggtitle(paste("boxplot of count for",cnames[i])))
}
# using library(gridExtra)
gridExtra::grid.arrange(gn1,gn2,gn3,gn4,gn5,ncol = 2)

# Loop to remove outliers by capping upperfence and lower fence values
for(i in cnames){
  print(i)
  #Quartiles
  Q1 = quantile(Bike_Rent[,i],0.25)
  Q3 = quantile(Bike_Rent[,i],0.75)
  
  #Inter quartile range 
  IQR = Q3-Q1
  
  # Upperfence and Lower fence values 
  UL = Q3 + (1.5*IQR(Bike_Rent[,i]))
  LL = Q1 - (1.5*IQR(Bike_Rent[,i]))
  
  # No of outliers and inliers in variables 
  No_outliers = length(Bike_Rent[Bike_Rent[,i] > UL,i])
  No_inliers = length(Bike_Rent[Bike_Rent[,i] < LL,i])
  
  # Capping with upper and inner fence values 
  Bike_Rent[Bike_Rent[,i] > UL,i] = UL
  Bike_Rent[Bike_Rent[,i] < LL,i] = LL
  
}

# Lets plot boxplots after removing outiers 
col = rainbow(ncol(Bike_Rent))
for(i in 1:length(cnames))
{
  assign(paste0("gn",i),ggplot(aes_string(y=(cnames[i]),x = 'count'),
                               data=subset(Bike_Rent))+
           stat_boxplot(geom = "errorbar",width = 0.5) +
           geom_boxplot(outlier.color = "red",fill=col[i],
                        outlier.shape = 18,outlier.size = 1,notch = FALSE)+
           theme(legend.position = "bottom")+
           labs(y = cnames[i],x='count')+
           ggtitle(paste("boxplot of count for",cnames[i])))
}

# using library(gridExtra)
gridExtra::grid.arrange(gn1,gn2,gn3,gn4,gn5,ncol = 2)

#### Data understanding using visualization  ###
####  Multi-Variate Analysis, Univariate Analysis and Bi-Variate Analysis of Train dataset   ###

###Univariate Analysis###
# Histogram for continuous variables to check  distribution of each variable 
col = rainbow(ncol(Bike_Rent))
for(i in 1:length(cnames))
{
  assign(paste0("h",i),ggplot(aes_string(x=(cnames[i])),
                              data=subset(Bike_Rent))+
           geom_histogram(fill= col[i],colour = "black",bins=30)+geom_density()+
           scale_y_continuous(breaks =pretty_breaks(n=10))+
           scale_x_continuous(breaks = pretty_breaks(n=10))+
           theme_bw()+xlab(cnames[i])+ylab("Frequency")+
           ggtitle(paste("Distribution of ",cnames[i])))
}

# using library(gridExtra)
gridExtra::grid.arrange(h1,h2,h3,h4,h5,ncol = 2)

# Lets check impact of continous variables on target variable
col = rainbow(ncol(Bike_Rent))
for(i in 1:length(cnames))
{
  assign(paste0("s",i),ggplot(aes_string(y='count',x = (cnames[i])),
                              data=subset(Bike_Rent))+
           geom_point(alpha=0.5,color=col[i]) +
           # labs(title = "Scatter Plot of count vs", x = (cnames[i]), y = "count")+
           ggtitle(paste("Scatter Plot:  count v/s",cnames[i])))
}

# using library(gridExtra)
gridExtra::grid.arrange(s1,s2,s3,s4,s5,ncol = 2)

## count vs temperature(atemp) : as temperature increase Bike rent count also increases 
## count vs humidity : humidity doesnt have any effect on bikerent count
## count vs windspeed : windspeed doesnt have any effect on bikerent count
## count vs count : please ignore this plot as it is our target variable

options(scipen = 999)

# Let us check impact of categorical variables on count
col = rainbow(ncol(Bike_Rent))
for(i in 1:length(cat_cnames))
{
  assign(paste0("b",i),ggplot(aes_string(y='count',x = (cat_cnames[i])),
                              data=subset(Bike_Rent))+
           geom_bar(stat = "identity",fill = col[i]) +
           labs(title = "Scatter Plot of count vs", x = (cnames[i]), y = "count")+
           ggtitle(paste("Number of bikes rented with respect to",cat_cnames[i])))+
    theme(axis.text.x = element_text( color="black", size=8))+
    theme(plot.title = element_text(face = "bold"))
}

# using library(gridExtra)
gridExtra::grid.arrange(b1,b2,b3,b4,ncol = 2)
gridExtra::grid.arrange(b5,b6,b7,ncol = 2)

# From barplot we can observe below points 
# Season:Bike rent count is high in season 3(fall) and low in season 1(springer)
aggregate(count ~ season ,sum,data = Bike_Rent)

# year : Bike rent count is high in year 1 (in 2012)
aggregate(count ~ year ,sum,data = Bike_Rent)

# month : Bike rent count is high in month of august and low in jan
aggregate(count ~ month,sum,data = Bike_Rent)

# holiday : Bike rent count is high on holidays ie 0
aggregate(count ~ holiday ,sum,data = Bike_Rent)

# weekday :From bar plot we can see maximum bikes rented on 5th day and least bikes on day 0.
aggregate(count ~ weekday ,sum,data = Bike_Rent)

# workingday : Bike rent count is high on working day  ie 1
aggregate(count ~ workingday,sum,data = Bike_Rent)

# weather : Bike rent count is high on weather 1: ie when the weather is 
# Clear, Few clouds, Partly cloudy, Partly cloudy
aggregate(count ~ weather,sum,data = Bike_Rent)

### Feature Selection ##

str(Bike_Rent)

# Using corrplot library we do correlation analysis for numeric variables
# Let us derive our correlation matrix 
Correlation_matrix = cor(Bike_Rent[,cnames])
Correlation_matrix

## By looking at correlation matrix we can say temperature and atemp are highly correlated (>0.99)

#Lets plot correlation plot using corrgram library 
corrgram(Bike_Rent[,cnames],order = F,upper.panel = panel.pie, diag.panel=panel.density, lower.panel=panel.cor,
         text.panel = panel.txt,main="Correlation Plot For Numeric Variables")


## Lets find significant categorical variables usig ANOVA test

# Anova analysis for categorical variable with target numeric variable
for(i in cat_cnames){
  print(i)
  Anova_result= summary(aov(formula = count~ Bike_Rent[,i],Bike_Rent))
  print(Anova_result)
}

## From the anova result, we can observe working day,weekday and holiday has p value > 0.05, 
## so delete these variables, Not to be consider in model

## Dimension reduction
Bike_Rent = subset(Bike_Rent,select = -c(atemp,holiday,weekday,workingday))


# Lets check dimensions after dimension reduction 
dim(Bike_Rent)

head(Bike_Rent)

# Lets check column names after dimension reduction 
names(Bike_Rent)

# Lets define/update  continous and categorical variables after dimension reduction
# Continuous variable
cnames= c('temperature','humidity', 'windspeed', 'count')

# Categorical variables
cat_cnames = c('season', 'year', 'month','weather')

### Feature Scaling   ###
## Since as it is mentioned in data dictionary the values of temp,humidity,windspeed variables
## are already normalized values so no need to go for feature scaling instead we will visualize the variables to see normality 

## Normality Check
# Normality check using normal qq plot
for(i in cnames){
  qqplot= qqnorm(Bike_Rent[,i])
  print(i)
}

## # Normality check using histogram plot(we already plotted hist in 
# data understanding)
gridExtra::grid.arrange(h1,h2,h3,h4,h5,ncol = 2)

#check summary of continuous variable to check the scaling- 
for(i in cnames){
  print(i)
  print(summary(Bike_Rent[,i]))
}

## From normal qq plot,histplot and by looking at summary of numeric variables we can say data is normally distributed
str(Bike_Rent)

###  Model Development   ##
# # Let's clean R Environment, as it uses RAM which is limited
# library(DataCombine)
# rmExcept("Bike_Rent")

## Lets convert all categorical variables to dummy variables
## As we can't pass categorical variables directly in to regression problems

# Lets save our preprocessed data into df data set 
df1 = Bike_Rent
Bike_Rent = df1

# Lets call Categorical varaibles after feature selection using ANOVA 
cat_cnames= c("season","year","month","weather")

# lets create dummy variables using dummies library

Bike_Rent = dummy.data.frame(Bike_Rent,cat_cnames)
dim(Bike_Rent)
head(Bike_Rent)
# we can see dummy variables are created in Bike rent dataset

# To avoid dummy variable trap I am hoing to remove 1 dummy variable form each type of categorical variable

Bike_Rent = subset(Bike_Rent,select=-c(season4,year1,month12,weather3))


# Divide data into train and test sets
set.seed(786)
train.index = createDataPartition(Bike_Rent$count, p = .80, list = FALSE)
train = Bike_Rent[ train.index,]
test  = Bike_Rent[-train.index,]

# Function for Error metrics to calculate the performance of model
mape= function(y,yhat){
  mean(abs((y-yhat)/y))*100
}

# Function for r2 to calculate the goodness of fit of model
rsquare=function(y,yhat){
  cor(y,yhat)^2
}

# Function for RMSE value 
rmse = function(y,yhat){
  difference = y - yhat
  root_mean_square = sqrt(mean(difference^2))
  print(root_mean_square)
}


### Multiple Linear Regression model  ###
# Lets build multiple linear regression model on train data 
# we will use the lm() function in the stats package
LR_Model = lm(count ~.,data = train)

# Check summary
summary(LR_Model) # Adjus.Rsquare = 0.833
# Lets check the assumptins of ols regression 
# 1) Error should follow normal distribution - Normal qqplot
# 2) No heteroscadacity - Residual plot
# 3) No multicolinearity between Independent variables 
# 4) No autocorrelation between errors
par(mfrow = c(2, 2))# Change the panel layout to 2 x 2
plot(LR_Model)


# Now Lets predict on test data 
LR_test= predict(LR_Model,test[-21])

# Lets check performance of model
# MAPE For test data
LR_MAPE_Test=mape(test[,21],LR_test)

# Rsquare For test data
LR_r2_test=rsquare(test[,21],LR_test)

# rmse For test data
LR_rmse_test = rmse(test[,21],LR_test)


# For test data 
print(postResample(pred = LR_test, obs =test$count))
print(paste("RMSE:",LR_rmse_test,"R2:",LR_r2_test))
print(paste("MAPE:",LR_MAPE_Test))
print(paste("LR_Accuracy:",(100-LR_MAPE_Test)))

############################
require(lmtest)
library(car)
dwtest(LR_Model)
####  Desicision Tree   ####

library(rpart)  			        # Popular decision tree algorithm
library(rattle)					      # Fancy tree plot
library(rpart.plot)				    # Enhanced tree plots
library(RColorBrewer)				  # Color selection for fancy tree plot

# Lets Build decision tree model on train data using rpart library 
DT_model= rpart(count~.,train,method = "anova")
DT_model

# Lets plot the decision tree model using rpart.plot library 
library(rpart.plot)	
rpart.plot(DT_model,type=2,digits=2,fallen.leaves=T,tweak = 1.2)


# Prediction on test data
DT_test= predict(DT_model,test[-21])


# MAPE For  test data
DT_MAPE_Test = mape(test[,21],DT_test)#28.09


# Rsquare For test data       
DT_r2_test = rsquare(test[,21],DT_test)#0.72


# rmse For test data
DT_rmse_test = rmse(test[,21],DT_test)
print(paste("RMSE:",DT_rmse_test,"R2:",DT_r2_test))
print(paste("MAPE:",DT_MAPE_Test))
print(paste("DT_Accuracy:",(100-DT_MAPE_Test)))

#### Random Search CV In Decision Tree  ###
# Lets set parameters to pass into our decision tree model 
# Lets use caret package for the same 
control = trainControl(method="repeatedcv", number=5, repeats=1,search='random')
maxdepth = c(1:30)
tunegrid = expand.grid(.maxdepth=maxdepth)

# Lets build a model using above parameters on train data 
RDT_model = caret::train(count~., data=train, method="rpart2",trControl=control,tuneGrid= tunegrid)
print(RDT_model)

# Lets look into best fit parameters
best_fit_parameters = RDT_model$bestTune
print(best_fit_parameters)

# Again rebuild decision tree model using randomsearch best fit parameter i.e., with maximum depth = 9
RDT_best_model = rpart(count~.,train,method = "anova",maxdepth=9)


# Prediction on test data 
RDT_test = predict(RDT_best_model,test[-21])

# Lets check Model performance on train data


# MAPE for test data 
RDT_MAPE_Test =mape(test[,21],RDT_test)


# Rsquare for test data
RDT_r2_test = rsquare(test[,21],RDT_test)


# rmse For test data
RDT_rmse_test = rmse(test[,21],RDT_test)

print(paste("RMSE:",RDT_rmse_test,"R2:",RDT_r2_test))
print(paste("MAPE:",RDT_MAPE_Test))
print(paste("RDT_Accuracy:",(100-RDT_MAPE_Test)))

### Grid Search CV in Decision Tree

control =trainControl(method="repeatedcv", number=5, repeats=2, search="grid")
tunegrid = expand.grid(.maxdepth=c(6:20))

# Lets build a model using above parameters on train data
GDT_model = caret::train(count~.,train, method="rpart2", tuneGrid=tunegrid, trControl=control)
print(GDT_model)

# Lets look into best fit parameters from gridsearch cv DT 
best_parameter = GDT_model$bestTune
print(best_parameter)

# Again rebuild decision tree model using gridsearch best fit parameter i.e., with maximum depth = 9
GDT_best_model = rpart(count ~ .,train, method = "anova", maxdepth =9)


# Prediction on test data 
GDT_test = predict(GDT_best_model,test[-21])


# Mape for test data using gridsearch cv  DT
GDT_MAPE_Test = mape(test[,21],GDT_test)


# Rsquare for test data
GDT_r2_test=rsquare(test[,21],GDT_test)


# rmse For test data
GDT_rmse_test = rmse(test[,21],GDT_test)

print(paste("RMSE:",GDT_rmse_test,"R2:",GDT_r2_test))
print(paste("MAPE:",GDT_MAPE_Test))
print(paste("GDT_Accuracy:",(100-GDT_MAPE_Test)))

#### Random Forest  ###

# Lets Build random forest model on train data using randomForest library 
RF_model= randomForest(count~.,train,ntree=100,method="anova")


# Prediction on test data
RF_test = predict(RF_model,test[-21])

# MAPE For test data
RF_MAPE_Test = mape(test[,21],RF_test)


# Rsquare For test data       
RF_r2_test=rsquare(test[,21],RF_test)


# rmse For test data
RF_rmse_test = rmse(test[,21],RF_test)

print(paste("RMSE:",RF_rmse_test,"R2:",RF_r2_test))
print(paste("MAPE:",RF_MAPE_Test))
print(paste("RF_Accuracy:",(100-RF_MAPE_Test)))

##### Random Search CV in Random Forest
control = trainControl(method="repeatedcv", number=5, repeats=1,search='random')
maxdepth = c(1:30)
tunegrid = expand.grid(.maxdepth=maxdepth)

# Lets build modelon train data using random search 
RRF_model = caret::train(count~., data=train, method="rf",trControl=control,tuneLength=10)
print(RRF_model)

# Best fit parameters
best_parameter = RRF_model$bestTune

print(best_parameter)

# RMSE was used to select the optimal model using the smallest value.
# The final value used for the model was mtry = 12.
#   mtry
# 2   12

# Lets build model based on best fit parameters
RRF_best_model = randomForest(count ~ .,train, method = "anova", mtry=12, importance=TRUE)


# Prediction on test data
RRF_test= predict(RRF_best_model,test[-21])


# Mape for test data
RRF_MAPE_Test = mape(test[,21],RRF_test) 


# Rsquare for test data
RRF_r2_test = rsquare(test[,21],RRF_test)


# rmse For test data
RRF_rmse_test = rmse(test[,21],RRF_test)

print(paste("RMSE:",RRF_rmse_test,"R2:",RRF_r2_test))
print(paste("MAPE:",RRF_MAPE_Test))
print(paste("RRF_Accuracy:",(100-RRF_MAPE_Test)))

#### Grid search CV in Random Forest

control = trainControl(method="repeatedcv", number=5, repeats=2, search="grid")
tunegrid = expand.grid(.mtry=c(6:20))

# Lets build a model using above parameters on train data using random forest grid search cv 
GRF_model= caret::train(count~.,train, method="rf", tuneGrid=tunegrid, trControl=control)
print(GRF_model)

# Best fit parameters
best_parameter = GRF_model$bestTune
print(best_parameter)

# RMSE was used to select the optimal model using the smallest value.
# The final value used for the model was mtry = 9.
#   mtry
# 4    9

# Build model based on Best fit parameters
GRF_best_model = randomForest(count ~ .,train, method = "anova", mtry=9)


# Prediction for test data
GRF_test= predict(GRF_best_model,test[-21])


# Mape calculation of test data
GRF_MAPE_Test = mape(test[,21],GRF_test)


# Rsquare for test data
GRF_r2_test=rsquare(test[,21],GRF_test)


# rmse For test data
GRF_rmse_test = rmse(test[,21],GRF_test)

print(paste("RMSE:",GRF_rmse_test,"R2:",GRF_r2_test))
print(paste("MAPE:",GRF_MAPE_Test))
print(paste("GRF_Accuracy:",(100-GRF_MAPE_Test)))

##### Gradient Boosting

library(gbm)

# Lets build a Gradient Boosting model for regression problem
GB_model = gbm(count~., data = train,distribution = "gaussian", n.trees = 100, interaction.depth = 2)


# Model Prediction on test data
GB_test = predict(GB_model, test[-21], n.trees = 100)


# Mape for test data
GB_MAPE_Test=mape(test[,21],GB_test)


# Rsquare for test data
GB_r2_test=rsquare(test[,21],GB_test)


# rmse For test data
GB_rmse_test = rmse(test[,21],GB_test)

print(paste("RMSE:",GB_rmse_test,"R2:",GB_r2_test))
print(paste("MAPE:",GB_MAPE_Test))
print(paste("GB_Accuracy:",(100-GB_MAPE_Test)))

#### Random Search CV in Gradient Boosting

control = trainControl(method="repeatedcv", number=5, repeats=1,search="random")
#maxdepth = c(1:30)
#tunegrid = expand.grid(.maxdepth=maxdepth)

# Model devlopment on train dat
RGB_model = caret::train(count~., data=train, method="gbm",trControl=control,tuneLength=10)

print(RGB_model)

# Best fit parameters
best_parameter = RGB_model$bestTune
print(best_parameter)

# RMSE was used to select the optimal model using the smallest value.
# The final values used for the model were n.trees = 4347, interaction.depth =
#  3, shrinkage = 0.0860172 and n.minobsinnode = 5.
#   n.trees interaction.depth  shrinkage n.minobsinnode
# 2    4347                 3 0.0860172              5

# Build model based on best fit
RGB_best_model = randomForest(count ~ .,train, method = "anova", n.trees=4347,
                              interaction.depth=3,shrinkage=0.0256,n.minobsinnode=5)

# Prediction on test data
RGB_test= predict(RGB_best_model,test[-21])


# Mape calculation of test data
RGB_MAPE_Test = mape(test[,21],RGB_test)


# Rsquare calculation for test data
RGB_r2_test=rsquare(test[,21],RGB_test)


# rmse For test data
RGB_rmse_test = rmse(test[,21],GRF_test)

print(paste("RMSE:",RGB_rmse_test,"R2:",RGB_r2_test))
print(paste("MAPE:",RGB_MAPE_Test))
print(paste("RGB_Accuracy:",(100-RGB_MAPE_Test)))

### Grid Search CV in Gradient Boosting

control = trainControl(method="repeatedcv", number=5, repeats=2, search="grid")
tunegrid = expand.grid(n.trees = seq(4347,4357, by = 2), #seq(2565,2575, by = 2),
                       interaction.depth = c(2:4), 
                       shrinkage = c(0.01,0.02),
                       n.minobsinnode = seq(2,8, by = 2))

# Model devlopment on train data
GGB_model= caret::train(count~.,train, method="gbm", tuneGrid=tunegrid, trControl=control)
print(GGB_model)

# Best fit parameters
best_parameter = GGB_model$bestTune
print(best_parameter)

# RMSE was used to select the optimal model using the smallest value.
# The final values used for the model were n.trees = 4349, interaction.depth =
#  2, shrinkage = 0.01 and n.minobsinnode = 2.
#   n.trees interaction.depth shrinkage n.minobsinnode
# 2    4349                 2      0.01              2

# Build model based on best fit
GGB_best_model = randomForest(count ~ .,train, method = "anova", n.trees = 4349,
                              interaction.depth = 2,shrinkage = 0.01,n.minobsinnode = 2)

# Prediction on test data
GGB_test= predict(GGB_best_model,test[-21])


# Mape for test data
GGB_MAPE_Test = mape(test[,21],GGB_test)


# Rsquare for test data
GGB_r2_test=rsquare(test[,21],GGB_test)


# rmse For test data
GGB_rmse_test = rmse(test[,21],GGB_test)

print(paste("RMSE:",GGB_rmse_test,"R2:",GGB_r2_test))
print(paste("MAPE:",GGB_MAPE_Test))
print(paste("GGB_Accuracy:",(100-GGB_MAPE_Test)))

#### Results

Model = c('Linear Regression','Decision Tree for Regression',
          'Random Search in Decision Tree','Gird Search in Decision Tree',
          'Random Forest','Random Search in Random Forest',
          'Grid Search in Random Forest','Gradient Boosting',
          'Random Search in Gradient Boosting',
          'Grid Search in Gradient Boosting')

Test_Accuracy = c((100-LR_MAPE_Test),(100-DT_MAPE_Test),(100-RDT_MAPE_Test),
                  (100-GDT_MAPE_Test),(100-RF_MAPE_Test),(100-RRF_MAPE_Test),(100-GRF_MAPE_Test),
                  (100-GB_MAPE_Test),(100-RGB_MAPE_Test),(100-GGB_MAPE_Test))


MAPE_Test = c(LR_MAPE_Test,DT_MAPE_Test,RDT_MAPE_Test,
              GDT_MAPE_Test,RF_MAPE_Test,RRF_MAPE_Test,GRF_MAPE_Test,
              GB_MAPE_Test,RGB_MAPE_Test,GGB_MAPE_Test)

Rsquare_Test = c(LR_r2_test,DT_r2_test,RDT_r2_test,GDT_r2_test,
                 RF_r2_test,RRF_r2_test,GRF_r2_test,GB_r2_test,
                 RGB_r2_test,GGB_r2_test)


Rmse_Test = c(LR_rmse_test,DT_rmse_test,RDT_rmse_test,GDT_rmse_test,
              RF_rmse_test,RRF_rmse_test,GRF_rmse_test,GB_rmse_test,
              RGB_rmse_test,GGB_rmse_test)

Final_results = data.frame(Model,Test_Accuracy,MAPE_Test,Rsquare_Test,Rmse_Test)

Final_results

# From above results Random Forest model have optimum values and this
# algorithm is good for our data 

# Lets save the out put of finalized model (Grid search RF)
Predicted_Bike_Rental_Count = predict(GRF_best_model,test[-21])
.
# Exporting the output to hard disk for further use
Final_test <- as.data.frame(cbind(test,Predicted_Bike_Rental_Count))

Final_output <- as.data.frame(cbind(Predicted_Bike_Rental_Count,Final_test$count))

names(Final_output)[2] <- "Actual_Bike_Rental_Count"

dim(Final_output)

Final_output <- Final_output[with(Final_output, order(Actual_Bike_Rental_Count)), ]
Final_output$Predicted_Bike_Rental_Count <- round(Final_output$Predicted_Bike_Rental_Count)
Final_output

# Finally, we designed a model, which predicts the Bike Rental Count.

# Exporting the output to hard disk for further use
write.csv(Final_output,"C:\\Users\\Sanjeev\\Desktop\\Edwisor\\Bike_rent_Prediction_2nd_Project\\Bike_Rental_Count_RF_results_R.csv")
write.csv(Final_results,"C:\\Users\\Sanjeev\\Desktop\\Edwisor\\Bike_rent_Prediction_2nd_Project\\Bike_Rental_Count_summary_R.csv")