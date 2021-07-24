########## HarvardX PH125.9x Project ##########
#Wine Quality Prediction

########## Preparing to work - Libraries and download data ##########

# Loading libraries

if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
if(!require(readr)) install.packages("readr", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("corrplot", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")


library(dplyr)
library(ggplot2)
library(tidyverse)
library(caret)
library(data.table)
library(knitr)
library(readr)
library(corrplot)
library(randomForest)


#Downloading the wine quality data dataset
#Source: https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009
#https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009/download

#CSV file is available on my GitHub
winedata_url <- "https://raw.github.com/drikarx/wine_quality/main/winequality-red.csv"

#Creating the data set with the Kaggle data
winedata <- read_csv(winedata_url)
winedata <- as.data.frame(winedata)

head(winedata)

winedata<- winedata %>% dplyr::rename('fixed_acidity'= 'fixed acidity', 'volatile_acidity'= 'volatile acidity', 'citric_acid'= 'citric acid',
  'residual_sugar'= 'residual sugar', 'free_sulfur_dioxide'= 'free sulfur dioxide', 'total_sulfur_dioxide'='total sulfur dioxide')

head(winedata)

winedata <- transform(winedata, quality = as.integer(quality))

save(winedata, file="winedata.RData")


########## Exploring the data ##########

class(winedata)
str(winedata)
dplyr::glimpse(winedata)
anyNA(winedata)
summary(winedata)

#We have 1599 observations and 12 variables in our dataset. Those 12 variables are:
colnames(winedata)


#Quality Distribution
barplot(table(winedata$quality),main="Quality Rates Distribution", border="black", col="cornflowerblue")

table(winedata$quality)


#Quality vs variables

attach(winedata)
par(mfrow=c(4,3))
plot(quality,`fixed_acidity`, main="Quality vs Fixed Acidity")
plot(quality,`volatile_acidity`, main="Quality vs Volatile Acidity")
plot(quality,`citric_acid`, main="Quality vs Citric Acid")
plot(quality,`residual_sugar`, main="Quality vs Residual Sugar")
plot(quality,`chlorides`, main="Quality vs Chlorides")
plot(quality,`free_sulfur_dioxide`, main="Quality vs Free Sulfur Dioxide")
plot(quality,`total_sulfur_dioxide`, main="Quality vs Total Sulfur Dioxide")
plot(quality,`density`, main="Quality vs Fixed Acidity")
plot(quality,`pH`, main="Quality vs Density")
plot(quality,`sulphates`, main="Quality vs Sulphates")
plot(quality,`alcohol`, main="Quality vs Alcohol")


#Correlation
cor(winedata)

correlation<- cor(winedata)


corrplot(correlation, type = "full", order = "hclust", 
         tl.col = "black", tl.srt = 45)

#Classifying the quality between good or bad

winedata<- winedata %>% mutate(badge = ifelse(quality >= 6, 'good', 'bad'))
winedata <- mutate(winedata, quality = as.factor(quality),badge = as.factor(badge))

str(winedata)

#Quality Badge distribution

barplot(table(winedata$badge),main="Badge Distribution", border="black", col="darkslategray2")

table(winedata$badge)


# The variables that will be used for predicting are all the variables but quality
winevar <- c("fixed_acidity", "volatile_acidity", "citric_acid",
          "residual_sugar", "chlorides", "free_sulfur_dioxide",
          "total_sulfur_dioxide", "density", "pH",
          "sulphates", "alcohol")


#Splitting thedata into training and test set
set.seed(1, sample.kind="Rounding")
train_index <- createDataPartition(y = winedata$badge, 
                                                 times = 1, 
                                                 p = 0.7, 
                                                 list = FALSE)

train <- winedata[train_index, ]
test <- winedata[-train_index, ]


anyNA(train)
str(train)
summary(train)


##### Prediction #####


# Linear Regression
# Predict the wine badge (good/bad) based on citric_acid + chlorides + alcohol

# Train the linear regression model
fit_lm <- train %>% 
  mutate(badge = ifelse(badge == "good", 1, 0)) %>%
  lm(badge ~ citric_acid + chlorides + alcohol, data = .)

# Predict
p_hat_lm <- predict(fit_lm, newdata = test)

# Convert the predicted value to factor
y_hat_lm <- factor(ifelse(p_hat_lm >= 1, "good", "bad"))

# Results
caret::confusionMatrix(y_hat_lm, test$badge)

result_lm <- caret::confusionMatrix(y_hat_lm, test$badge)


# Random Forest
# Predict the wine badge (good/bad) based on all the variables

#Formula
fml <- as.formula(paste("badge", "~", 
                        paste(winevar, collapse=' + ')))

# Train the model
fit_rf <- randomForest(formula = fml, data = train)

# Predict
y_rf <- predict(object = fit_rf, newdata = test)

# Results
caret::confusionMatrix(data = y_rf, 
                       reference = test$badge, 
                       positive = "good")

result_rf <- caret::confusionMatrix(data = y_rf, 
                                    reference = test$badge, 
                                    positive = "good")

# Plot the error curve
data.frame(fit_rf$err.rate) %>% mutate(x = 1:500 ) %>% 
  ggplot(aes(x = x)) + 
  geom_line(aes(y = good),   col = "mediumseagreen",) +
  geom_line(aes(y = bad), col = "indianred3") +
  ggtitle("Random Forest Error Curve") +
  ylab("Error") +
  xlab("Number of trees")

# Variable importance plot
varImpPlot(fit_rf, main = "Random Forest Variable importance")


## Result compasison

result_lm$overall["Accuracy"]

result_rf$overall["Accuracy"]
