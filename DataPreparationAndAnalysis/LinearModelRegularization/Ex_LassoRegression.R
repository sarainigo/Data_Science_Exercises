set.seed(123)
library(caret)
library(corrplot)
library(dvmisc)
library(tidyverse)
library(broom)
library(glmnet)
library(doMC)

data = data.frame(swiss)
y <- data$Fertility

# split the data
train_test<-createDataPartition(y,p=0.8,list = FALSE)
data_train<-data[train_test,]
data_test<-data[-train_test,]

# linear model
model<-lm(Fertility~.,data = data_train)
summary(model)

# lasso regression
y <- data_train$Fertility
x <- model.matrix(Fertility~.,data_train)[,-1]
lambdas <- 10^seq(3, -2, length.out=100)

fit <- glmnet(x,y, alpha = 1, lambda = lambdas)
summary(fit)

# find min lambda 
cv_fit <- cv.glmnet(x, y, alpha = 1, lambda = lambdas)

# min lambda
opt_lambda <- cv_fit$lambda.min
opt_lambda

# plot lambda vs MSE
plot(cv_fit)


# out-of-sample performance
y_test <- data_test$Fertility
x_test<- model.matrix(Fertility~.,data_test)[,-1]

# lm
prediction_lm<-predict(model, data.frame(x_test))
MES_lm=mean((prediction_lm-y_test)^2)

# lasso
fit <- cv_fit$glmnet.fit
prediction_lasso<-predict(fit,s=opt_lambda,newx = x_test)
MES_lasso=mean((prediction_lasso-y_test)^2)

# coefficients
model_2<-glmnet(x,y,alpha=1,lambda = opt_lambda)
coefs_2=coef(model_2)
coefs_2


