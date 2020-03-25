set.seed(123)
library(e1071)
library(DMwR)
library(MLmetrics)

col_names <- c("edibility","cap-shape","cap-surface","cap-color","bruises","odor","gill-attachment","gill-spacing","gill-size","gill-color","stalk-shape","stalk-root","stalk-surface-above-ring","stalk-surface-below-ring","stalk-color-above-ring","stalk-color-below-ring", "veil-type","veil-color","ring-number","ring-type","spore-print-color","population","habitat")
data <- data.frame(read.csv(url("https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"), header=FALSE, sep=",",col.names = col_names))

data[data == '?'] <- NA
#checking missing values in all columns
missing_val=colSums(is.na(data))
is_na=is.na(data)
# total number of missing values. There are 2480 NA
n_missing=sum(is_na)
# delete observations which have missing values
data_new <- na.omit(data)
# split data 80-20
train_test <- sample.int(n = nrow(data_new), size = floor(0.8*nrow(data_new)), replace = F)
train <- data_new[train_test, ]
test  <- data_new[-train_test, ]

# Naive Bayes classifier
nbayes_model<-naiveBayes(train$edibility~., train)

# accuracy x=predicted data y=observed data
pred <- predict(nbayes_model, test, type = "class")
test_accuracy=Accuracy(pred, test$edibility)

pred_train <- predict(nbayes_model, train, type = "class")
train_accuracy=Accuracy(pred_train, train$edibility)

# confusion matrix 
table(pred, test$edibility,dnn=c("Predicted","Actual"))


