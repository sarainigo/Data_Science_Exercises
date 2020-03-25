set.seed(123)
library(caret)
library(corrplot)
library(dvmisc)
library(Metrics)

#download data
col_names <- c("Longitudinal position","Prismatic coefficient", "Length-displacement ratio", 
               "Beam-draught ratio", "Length-beam ratio", "Froude number", "Residuary resistance")
data <- read.csv(url("http://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data"), 
                 header=FALSE, col.names = col_names, sep="", as.is=TRUE)
yatch_data = data.frame(data)

#split the data
train_test<-createDataPartition(yatch_data$Residuary.resistance,p=0.8,list = FALSE)
head(train_test)  
data_train<-yatch_data[train_test,]
data_test<-yatch_data[-train_test,]

#linear model
model<-lm(Residuary.resistance~.,data = data_train)
training_MSE = get_mse(model, var.estimate = FALSE)
training_RMSE = sqrt(training_MSE)
training_R2 = summary(model)$r.squared

# define training control
train_control <- trainControl(method="boot", number=1000)
# train the model
model_2 <- train(Residuary.resistance~., data=yatch_data, trControl=train_control, method="lm")

# mean RMSE and R2 for the fit
print(model_2)
# histogram of the RMSE values
histogram(model_2,metric="RMSE")

#MSEtest of original model and bootstrap model
x_test <- data_test[,1:6]
y_test <- data_test[,7]

# original model
predictions <- predict(model, x_test)
test_MSE = mean((predictions-y_test)^2)
test_MSE
# bootstrap model
predictions_2 <- predict(model_2, x_test)
test_MSE_2 = mean((predictions_2-y_test)^2)

# COMPARISON
test_MSE
test_MSE_2


