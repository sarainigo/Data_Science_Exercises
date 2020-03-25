set.seed(123)

data <- read.table("~/RStudioProjects/Statistics/HW4/cement.txt", sep="",header = TRUE)

# start with no predictor in the model
model_0<-lm(y~1,data = data)

add1(model_0,~.+x1+x2+x3+x4,test = 'F')
# Since the smallest value corresponds to x4, we should add x4 to the model

model_x4<-lm(y~x4,data=data)

add1(model_x4,~.+x1+x2+x3,test = 'F')
# We add x1 to the model

model_x4x1<-lm(y~x1+x4,data=data)

drop1(model_x4x1,~.,test='F')
# As we can see, both x1 and x4 should stay in the model

add1(model_x4x1,~.+x2+x3,test='F')

# Since the smallest value corresponds to x2, we should add x2 to the model
model_x4x1x2<-lm(y~x1+x2+x4,data = data)

drop1(model_x4x1x2,~.,test = 'F')
# As we can see, x4 must be removed from the model

model_x1x2<-lm(y~+x1+x2,data=data)

add1(model_x1x2,~.+x3+x4,test = 'F')
# We can see that no more predictors can be added to the model

