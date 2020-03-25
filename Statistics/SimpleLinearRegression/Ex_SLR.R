set.seed(123)

n= 48

skincancer <- read.csv("~/RStudioProjects/skincancer.txt", sep="")
x = skincancer$Lat
y = skincancer$Mort

x_mean = mean(x)
y_mean = mean(y)

# Computation of b1
b1_n = sum((x-x_mean)*(y-y_mean))
b1_d = sum((x-x_mean)^2)
b1 = b1_n/b1_d
b0 = y_mean - b1*x_mean
yhat = b0 + b1*x


# Computation of SST, SSR, SSE
SST = sum((y-y_mean)^2)
SSE = sum((y-yhat)^2)
SSR = sum((yhat-y_mean)^2)

# Computation of 95 confidence interval
MSE = SSE/(n-2)
s_e = sqrt(MSE/sum((x-x_mean)^2))
t = qt(0.975, n-2)

conf_int1 = b1 - t*s_e
conf_int2 = b1 + t*s_e

# VERIFICATION OF THE RESULTS
skincancer_fit = lm(Mort~Lat, data=skincancer)
summary(skincancer_fit)
confint(skincancer_fit)
plot(skincancer_fit)
  
  
