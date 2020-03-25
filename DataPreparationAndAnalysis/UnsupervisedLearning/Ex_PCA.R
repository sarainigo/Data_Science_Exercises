# PROBLEM 2

col_names<-c("Type","Alcohol","Malic Acid","Ash","Alcalinity of ash","Magnesium",
             "Total phenols","Flavanoids","Nonflavanoid phenols","Proanthocyanins",
             "Color intensity","Hue","OD280/OD315 of diluted wines","Proline")
data <- data.frame(read.csv(url("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"), 
                            header=FALSE))

# first, examine the data
apply(data , 2, mean)
apply(data , 2, var) 

# We should scale the data, since we have very different means and variances
pr_out<-prcomp(data , scale=TRUE)

# biplot of the results
biplot(pr_out , scale=0)


# standard deviations
pr_out$sdev

# variance explained by each principal component
pr_var <- pr_out$sdev^2
pr_var

# proportion of variance explained by each principal component
pve=pr_var/sum(pr_var)
pve

# screeplot
plot(pve , xlab="Principal Component ", ylab="Proportion of Variance Explained ", ylim=c(0,1),type='b') 
# cumulative proportion of variance explained
plot(cumsum(pve), xlab="Principal Component ", ylab=" Cumulative Proportion of Variance Explained ", ylim=c(0,1), type='b')


