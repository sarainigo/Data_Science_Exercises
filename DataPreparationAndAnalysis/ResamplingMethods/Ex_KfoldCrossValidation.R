set.seed(123)
library(caret)
library(corrplot)
library(dvmisc)

#download data
data <- read.csv(url("https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data-numeric"), 
                 header=FALSE, sep="")
data = data.frame(data)

y = data$V25

# split the data
train_test<-createDataPartition(y,p=0.8,list = FALSE)
data_train<-data[train_test,]
data_test<-data[-train_test,]

# fit logistic model
data_train$V25=as.factor(data_train$V25)
model<-glm(data_train$V25~., family=binomial(), data = data_train)
summary(model)

# Training Precision/Recall and F1 results
fitted = model$fitted.values
fitted_bin = rep(1, length(fitted))
fitted_bin[fitted>=0.5] <- 2
fitted_bin = as.factor(fitted_bin)
conf_mat_train = confusionMatrix(fitted_bin,data_train$V25)
conf_mat_train$byClass[5]
conf_mat_train$byClass[6]
conf_mat_train$byClass[7]

# k=10 fold cross-validation fit:
train_control <- trainControl(method="cv", number=10)
model_2 <- train(V25~., data=data, trControl=train_control, method="glm")
# summarize results
print(model_2)

# Training Precision/Recall and F1 results for model 2
x = data[,1:24]
fitted_2 = predict(model_2,x)
fitted_2 = as.factor(fitted_2)
data$V25=as.factor(data$V25)
conf_mat_train_2 = confusionMatrix(fitted_2,data$V25)
conf_mat_train_2$byClass[5]
conf_mat_train_2$byClass[6]
conf_mat_train_2$byClass[7]

# test comparison
x_test <- data_test[,1:24]
y_test <- data_test[,25]

# model glm
prediction = predict(model,x_test, type = "response")
prediction_bin = rep(1, length(prediction))
prediction_bin[prediction>=0.5] <- 2
prediction_bin = as.factor(prediction_bin)
data_test$V25 = as.factor(data_test$V25)
conf_mat_test = confusionMatrix(prediction_bin,data_test$V25)
conf_mat_test$byClass[5]
conf_mat_test$byClass[6]
conf_mat_test$byClass[7]

# model cross-validation
prediction_2 = predict(model_2,x_test)
prediction_2 = as.factor(prediction_2)
conf_mat_test = confusionMatrix(prediction_2,data_test$V25)
conf_mat_test$byClass[5]
conf_mat_test$byClass[6]
conf_mat_test$byClass[7]



