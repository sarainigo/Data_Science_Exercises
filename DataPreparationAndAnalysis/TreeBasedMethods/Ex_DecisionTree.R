# PROBLEM 1
n=100

label <-c(rep("Black",n))
feature = rnorm(n,5,2)
df_b <- data.frame(label,feature)

label <- c(rep("White",n))
feature = rnorm(n,-5,2)
df_w <- data.frame(label,feature)

df<-rbind(df_b,df_w)
rows <- sample(nrow(df))
df<- df[rows, ]

library(rpart)
library(rpart.plot)
fit <- rpart(label~., data = df, method = 'class')
rpart.plot(fit, extra = 106)
summary(fit)

# Node 1
p1 = 0.5
p2 = 0.5
gini_node1 <- p1 * (1 - p1) + p2 * (1 - p2)
entropy_node1 <- - (p1 * log(p1) + p2 * log(p2))

# Node 2 - predicted: Black
p1 = 0.99
p2 = 0.01
gini_node2 <- p1 * (1 - p1) + p2 * (1 - p2)
entropy_node2 <- - (p1 * log(p1) + p2 * log(p2))

# Node 3 - predited: White
p1 = 0
p2 = 1
gini_node3 <- p1 * (1 - p1) + p2 * (1 - p2)
entropy_node3 <- - (p1 * log(p1) + p2 * log(p2))

# Repeat experiment for othe normal

n=100

label <-c(rep("Black",n))
feature = rnorm(n,1,2)
df_b <- data.frame(label,feature)

label <- c(rep("White",n))
feature = rnorm(n,-1,2)
df_w <- data.frame(label,feature)

df<-rbind(df_b,df_w)
rows <- sample(nrow(df))
df<- df[rows, ]

library(rpart)
library(rpart.plot)
fit <- rpart(label~., data = df, method = 'class')
rpart.plot(fit, extra = 106)
summary(fit)


