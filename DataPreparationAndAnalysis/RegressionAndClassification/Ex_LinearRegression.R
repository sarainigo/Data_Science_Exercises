
set.seed(123)
library(MASS)
library(ggplot2)
#load of Boston sample
data= data.frame(Boston)

x = data$lstat
y = data$medv
# fit the model
model <- lm(y ~ x, data=data)
summary(model)

# plot of the resulting fit
p1 = ggplot(data, aes(x = x, y = y))
p1 <- p1 + geom_point()
p1 <- p1 + stat_smooth(method = "lm", col = "red")
p1 <- p1 + labs(x = "lstat", y="medv", title="lstat vs. medv plot")
p1

# plot fitted values vs residuals
p2<-ggplot(model, aes(.fitted, .resid))
p2 <- p2 + geom_point()
p2 <- p2 + stat_smooth(method="loess")+geom_hline(yintercept=0, col="red", linetype="dashed")
p2 <- p2 + labs(x = "fitted values", y="residuals", title="fitted values vs. residuals plot")
p2

# prediction of 5, 10, 15 with C.I and P.I
CI_5 <- predict(model, newdata=data.frame(x=5), interval="confidence", level = 0.95)
PI_5 <- predict(model, newdata=data.frame(x=5), interval="prediction", level = 0.95)

CI_10 <- predict(model, newdata=data.frame(x=10), interval="confidence", level = 0.95)
PI_10 <- predict(model, newdata=data.frame(x=10), interval="prediction", level = 0.95)

CI_15 <- predict(model, newdata=data.frame(x=15), interval="confidence", level = 0.95)
PI_15 <- predict(model, newdata=data.frame(x=15), interval="prediction", level = 0.95)


# modification of the regression to include lstat^2
model2<-lm(y~poly(x,2))
#comparison of R squared
R_2_model=summary(model)$r.squared
R_2_model2=summary(model2)$r.squared

# plot of the resulting fit
p3 = ggplot(data, aes(x = x, y = y))
p3 <- p3 + geom_point()
p3 <- p3 + stat_smooth(method = "lm", formula = y ~ x + I(x^2), col = "red")
p3 <- p3 + labs(x = "lstat", y="medv", title="lstat vs. medv plot")
p3

