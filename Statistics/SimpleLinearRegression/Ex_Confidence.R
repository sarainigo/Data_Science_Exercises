set.seed(123)
n=522
# Download the data
data <- read.csv("~/RStudioProjects/Statistics/HW2/APPENC07.txt", sep="", header=FALSE)

sales = data[sample(nrow(data), 100), ]
sales = data.frame(sales)

y = sales$V2
x = sales$V3

#Computation of SLR
b1_n = sum((x-x_mean)*(y-y_mean))
b1_d = sum((x-x_mean)^2)
b1 = b1_n/b1_d
b0 = y_mean - b1*x_mean
yhat = b0 + b1*x

# Computation of means
x_mean = mean(x)
y_mean = mean(y)

SSE = sum((y-yhat)^2)


# SLR comupted with funcion lm
model <- lm(V2 ~ V3, data=sales)
summary(model)
# We compute summary to observe the min and max values. min=1198, max=4746
summary(x)

# 95 Confidence Interval (CI) for regression line

newx <- seq(1198, 4746, by=0.05)
plot(x, y, xlab="Finished square feet", ylab="Sales price", main="Regression Plot")
abline(model, col="lightblue")

conf_interval <- predict(model, newdata=data.frame(V3=newx), interval="confidence",level = 0.95)
lines(newx, conf_interval[,2], col="blue", lty=2)
lines(newx, conf_interval[,3], col="blue", lty=2)

# 95 Prediction Interval (PI) for regression line

pred_interval <- predict(model, newdata=data.frame(V3=newx), interval="prediction",level = 0.95)
lines(newx, pred_interval[,2], col="orange", lty=2)
lines(newx, pred_interval[,3], col="orange", lty=2)


# 95 Confidence Interval (CI) at x=x_mean

conf_interval_xmean <- predict(model, newdata=data.frame(V3=x_mean), interval="confidence", level = 0.95)
conf_interval_xmean

# 95 Prediction Interval (PI) at x=x_mean

pred_interval_xmean <- predict(model, newdata=data.frame(V3=x_mean), interval="prediction", level = 0.95)
pred_interval_xmean

