set.seed(123)
library(caret)
library(corrplot)
library(dvmisc)
library(tidyverse)
library(broom)
library(glmnet)
library(doMC)

data = data.frame(mtcars)
y <- data$mpg
# split the data
train_test<-createDataPartition(y,p=0.8,list = FALSE)
data_train<-data[train_test,]
data_test<-data[-train_test,]

# linear model
model<-lm(mpg~.,data = data_train)
summary(model)

# ridge regression
y <- data_train$mpg
x <- data_train %>% select(cyl, disp, hp, drat, wt, qsec, vs, am, gear, carb) %>% data.matrix()
lambdas <- 10^seq(3, -2, length.out=100)

fit <- glmnet(x, y, alpha = 0, lambda = lambdas)
summary(fit)

# find min lambda 
cv_fit <- cv.glmnet(x, y, alpha = 0, lambda = lambdas)
# plot lambda vs MSE
plot(cv_fit)

# min lambda
opt_lambda <- cv_fit$lambda.min
opt_lambda

# out-of-sample performance
y_test <- data_test$mpg
x_test<- model.matrix(mpg~.,data_test)[,-1]

# lm
prediction_lm<-predict(model, data.frame(x_test))
MES_lm=mean((prediction_lm-y_test)^2)

# ridge
fit <- cv_fit$glmnet.fit
prediction_ridge<-predict(fit,s=opt_lambda,newx = x_test)
MES_ridge=mean((prediction_ridge-y_test)^2)

# coefficients
model_2<-glmnet(x,y,alpha=0,lambda = opt_lambda)
coefs_2=coef(model_2)
coefs_2


