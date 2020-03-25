set.seed(123)

# Generation of the dataset
n=40
x=runif(n,-1,1)
e=rnorm(n,0,0.01)
y=2*x+e

x_mean = mean(x)
y_mean = mean(y)

# Generation of regressor 1 through the origin
brto_n = sum(x*y)
brto_d = sum(x^2)
brto = brto_n/brto_d

yhat_1 = brto*x

# Generation of regresor 2 ordinary linear
b1_n = sum((x-x_mean)*(y-y_mean))
b1_d = sum((x-x_mean)^2)
b1 = b1_n/b1_d
b0 = y_mean - b1*x_mean

yhat_2 = b0 + b1*x

# Computation of residue and r2
SST_1 = sum((y-y_mean)^2)
SST_2 = sum((y-y_mean)^2)
SSR_1 = sum((yhat_1-y_mean)^2)
SSR_2 = sum((yhat_2-y_mean)^2)

r2_1 = SSR_1/SST_1
r2_2 = SSR_2/SST_2
res_1 = sum(y - yhat_1)
res_2 = sum(y - yhat_2)

