# PROBLEM 4

data <- data.frame(read.csv(url("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"), 
                            header=TRUE, sep=";"))

# delete target variable
data$quality <- NULL

# first, examine the data
apply(data , 2, mean)
apply(data , 2, var) 

# scale the data

data_scaled<-data.frame(scale(data))

# hierarchical clustering
hc_complete=hclust(dist(data_scaled), method="complete")
hc_single=hclust(dist(data_scaled), method="single")

# plot the dendrograms obtained 
par(mfrow=c(1,2))
plot(hc_complete, main="Complete Linkage")
plot(hc_single, main="Single Linkage")


# obtain two clusters by cutree
groups_2_complete = cutree(hc_complete,2)
table(groups_2_complete)
summary(groups_2_complete)

groups_2_single = cutree(hc_single,2)
table(groups_2_single)
summary(groups_2_single)




