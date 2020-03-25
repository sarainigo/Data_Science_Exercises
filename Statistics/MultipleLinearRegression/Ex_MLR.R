set.seed(123)
library(matlib)

col_names <- c("Y", "X1")
data <- data.frame(read.csv("~/RStudioProjects/Statistics/HW3/Copier.txt", sep="", header=FALSE, col.names = col_names))
model_data <- data.frame(read.csv("~/RStudioProjects/Statistics/HW3/Model.txt", sep="", header=FALSE))
data$X2 <- model_data$V1


# total number of minutes spent by the service person
Y = data$Y
# number of copiers serviced
X1 = data$X1
# binary predictor variable that indicates whether the copier model is small (X2=1) or large (X2=0)
X2 = data$X2

# QUESTION 1
# fit the model
model <- lm(Y ~ X1 + X2, data=data)
summary(model)
coefs = coefficients(model)
# estimated regression function
Yhat = coefs[1]+coefs[2]*X1+coefs[3]*X2

# QUESTION 2
n=45
p=2
X = cbind(X1,X2)
Y_mean = mean(Y)
SST = sum((Y-Y_mean)^2)
SSE = sum((Y-Yhat)^2)
SSR = sum((Yhat-Y_mean)^2)

MSE = SSE/(n-p-1)
s_e = sqrt(MSE*(t(t(X2))*solve(t(X)*X)*X2))
t = qt(0.975, n-p-1)

conf_int1 = Y_mean  - t*s_e
conf_int2 = Y_mean  + t*s_e


# QUESTION 3
res = resid(model)

plot(res ~ I(X1 * X2), data, xlab = "x1x2", ylab = "Residual")
title("Residual Plot against x1x2")

# QUESTION 4
# we are going to create a new model
data$X1X2 <- data$X1 * data$X2
# fit the model
model2 <- lm(Y ~ X1 + X2 + X1X2, data=data)
summary(model2)
coefs2 = coefficients(model2)
# estimated regression function
Yhat2 = coefs2[1]+coefs2[2]*X1+coefs2[3]*X2+coefs2[4]*X1*X2


