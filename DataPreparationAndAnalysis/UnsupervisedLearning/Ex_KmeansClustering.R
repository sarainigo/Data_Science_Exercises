# PROBLEM 3

data<-data.frame((USArrests))

# Examine the data
apply(data , 2, mean)
apply(data , 2, var)

data_scaled<-data.frame(scale(data))

apply(data_scaled , 2, mean)
apply(data_scaled , 2, var)

# K-means clustering for K from 2 to 10 
set.seed(123)
K_2 <- kmeans(data_scaled, centers = 2, nstart = 20)
K_3 <-kmeans(data_scaled, centers = 3, nstart = 20)
K_4 <-kmeans(data_scaled, centers = 4, nstart = 20)
K_5 <-kmeans(data_scaled, centers = 5, nstart = 20)
K_6 <-kmeans(data_scaled, centers = 6, nstart = 20)
K_7 <-kmeans(data_scaled, centers = 7, nstart = 20)
K_8 <-kmeans(data_scaled, centers = 8, nstart = 20)
K_9 <-kmeans(data_scaled, centers = 9, nstart = 20)
K_10 <-kmeans(data_scaled, centers = 10, nstart = 20)  

# within-cluster sum of squares for each value of k
K_2$tot.withinss
K_3$tot.withinss
K_4$tot.withinss
K_5$tot.withinss
K_6$tot.withinss
K_7$tot.withinss
K_8$tot.withinss
K_9$tot.withinss
K_10$tot.withinss


pkgs <- c("factoextra",  "NbClust")
install.packages(pkgs)

library(factoextra)
library(NbClust)

# Elbow method
fviz_nbclust(data_scaled, kmeans, method = "wss") +
  geom_vline(xintercept = 7, linetype = 2)+
  labs(subtitle = "Elbow method")

# plot optimal cluster
fviz_cluster(K_7, data = data_scaled)

