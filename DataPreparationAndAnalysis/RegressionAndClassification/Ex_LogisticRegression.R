set.seed(123)
library(caret)
library(corrplot)

col_names <- c("Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight", "Rings")
data <- data.frame(read.csv(url("https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"), header=FALSE, sep=",", col.names = col_names))

# remove Infant observations
data_new <- data.frame(droplevels( data[-which(data$Sex == "I"), ] ))

# test-train split
train_test<-createDataPartition(data_new$Sex,p=0.8,list = FALSE)
head(train_test)  
data_train<-data_new[train_test,]
data_test<-data_new[-train_test,]

# logistics regression model
model<-glm(data_train$Sex~.,family=binomial(),data = data_train)
summary(model)
confint(model)

# Confusion Matrix
prediction<-predict(model,newdata = data_test, interval="confidence")
prediction.dt<-ifelse(prediction>0.50, "M","F")
Pred <- as.factor(prediction.dt)
Predicted <- ordered(Pred, levels = c("M", "F"))
Actual <- ordered(data_test$Sex,levels = c("M", "F"))
conf_mat <-confusionMatrix(table(Predicted,Actual))
conf_mat

# ROC
predict <-predict(model,newdata = data_test)
prediction <- prediction(predict,data_test$Sex)
perf <- performance(prediction,"tpr","fpr")
plot(perf)

# We calculate the corr matrix
predictors<-data_new[,-1]
corr_mat <- cor(predictors)
corrplot(corr_mat,type = "upper", method = "number", diag = TRUE, tl.col = "blue")

