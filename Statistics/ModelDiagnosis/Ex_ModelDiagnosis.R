set.seed(123)
library(car)
library(olsrr) 

col_names <- c("Identification number","County","State","Land area","Total population","Percent of population 18-34",
               "Percent of population 65","Number of active physicians","Number of hospital beds","Total serious crimes",
               "Percent high school graduates","Percent bachelor's degrees","Percent below poverty level",
               "Percent unemployment", "Per capita income","Total personal income", "Geographic region")
data <- data.frame(read.csv("~/RStudioProjects/Statistics/HW4/CDI.txt", header=FALSE, sep="",col.names = col_names))

X1 = data$Percent.of.population.18.34
X2 = data$Number.of.active.physicians
X3 = data$Number.of.hospital.beds
X4 = data$Percent.below.poverty.level
X5 = data$Percent.unemployment
X6 = data$Per.capita.income
Y = data$Total.serious.crimes

# QUESTION 1: Variance inflation factors

# We need to regress each predictor on the remaining predictors
# predictor 1
model_1 = lm(X1~X2+X3+X4+X5+X6, data=data)
r2_1 = summary(model_1)$r.squared
VIF_1 = 1/(1-r2_1)

# predictor 2
model_2 = lm(X2~X1+X3+X4+X5+X6, data=data)
r2_2 = summary(model_2)$r.squared
VIF_2 = 1/(1-r2_2)

# predictor 3
model_3 = lm(X3~X2+X1+X4+X5+X6, data=data)
r2_3 = summary(model_3)$r.squared
VIF_3 = 1/(1-r2_3)

# predictor 4
model_4 = lm(X4~X2+X3+X1+X5+X6, data=data)
r2_4 = summary(model_4)$r.squared
VIF_4 = 1/(1-r2_4)

# predictor 5
model_5 = lm(X5~X2+X3+X4+X1+X6, data=data)
r2_5 = summary(model_5)$r.squared
VIF_5 = 1/(1-r2_5)

# predictor 6
model_6 = lm(X6~X2+X3+X4+X5+X1, data=data)
r2_6 = summary(model_6)$r.squared
VIF_6 = 1/(1-r2_6)

# other way
model = lm(Y~X1+X2+X3+X4+X5+X6, data=data)
vif = vif(model)

# predictor 2 and predictor 3 have substantial collinearity

# QUESTION 2: Studentized deleted residuals
model = lm(Y~X1+X2+X3+X4+X5+X6, data=data)
stud_del_resid=rstudent(model)
ols_plot_resid_stud(model)

outlier<-c()
j=1
for (i in (1:440)){
  if (abs(stud_del_resid[i])>3){
    outlier[j]=i
    j=j+1
  } 
}

# there are outliers in observations 6, 11, 15, 17 and 123

# QUESTION 3: Diagonal elements of the hat matrix

n = 440
p = 6

X<-data.matrix(data.frame(X1,X2,X3,X4,X5,X6))
hat_matrix<-X%*%(solve(t(X)%*%X))%*%t(X)
hat_diag<-diag(hat_matrix)
h_bar = (p+1)/n
extreme_xi<-c()
j=1
for (i in (1:n)){
  if ((3*h_bar)<hat_diag[i]){
    extreme_xi[j]=i
    j=j+1
  } 
}

# extreme_xi are the indexes of the xi extremes: 1   2   3   6   8  12  19  32  48  95 128 188 206 303 337 363 404 415

# QUESTION 4: DFFITS, DFBETAS and Cook’s distance values

# DFFIT
thr = 2*((sqrt(p+1))/(n-p-1))
# outlying with respect to their X values
dffits(model) [2]
if (abs(dffits(model) [2])>thr){print("Case 2 influential")} 
dffits(model) [8]
if (abs(dffits(model) [8])>thr){print("Case 8 influential")} 
dffits(model) [48]
if (abs(dffits(model) [48])>thr){print("Case 48 influential")} 
dffits(model) [128]
if (abs(dffits(model) [128])>thr){print("Case 128 influential")} 
dffits(model) [206]
if (abs(dffits(model) [206])>thr){print("Case 206 influential")} 
dffits(model) [404]
if (abs(dffits(model) [404])>thr){print("Case 404 influential")} 
# outlying with respect to their Y values
dffits(model) [2]
if (abs(dffits(model) [2])>thr){print("Case 2 influential")} 
dffits(model) [6]
if (abs(dffits(model) [6])>thr){print("Case 6 influential")} 

# in terms of DFFIT, all cases are influential

# DFBETAS
thr_b = 2/sqrt(n)
# outlying with respect to their X values
dfbetas(model)[2,] 
dfbetas(model)[8,]
dfbetas(model)[48,]
dfbetas(model)[128,]
dfbetas(model)[206,]
dfbetas(model)[404,]
# outlying with respect to their Y values
dfbetas(model)[2,]
dfbetas(model)[6,]

# check influence
abs(dfbetas(model)[2,])>thr_b
abs(dfbetas(model)[8,])>thr_b
abs(dfbetas(model)[48,])>thr_b
abs(dfbetas(model)[128,])>thr_b
abs(dfbetas(model)[206,])>thr_b
abs(dfbetas(model)[404,])>thr_b
abs(dfbetas(model)[6,])>thr_b

# in terms of DFBETAS, Cases 2 and 206 are influential to some parameters. Cases 8, 48, 128, 404 are not influential. Case 6 is influential.

# Cooks distance

cooks_d = cooks.distance(model)
# outlying with respect to their X values
cooks_d[2]
cooks_d[8]
cooks_d[48]
cooks_d[128]
cooks_d[206]
cooks_d[404]
# outlying with respect to their Y values
cooks_d[2]
cooks_d[6]

# in terms of cook's distance, Case 6 is influential
