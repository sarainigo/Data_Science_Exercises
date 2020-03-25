set.seed(123)

#boxplot

iris_data = data.frame(iris)
boxplot(iris_data[1:4])

#standard deviation

sd_sepall=sd(iris_data$Sepal.Length)
sd_sepalw=sd(iris_data$Sepal.Width)
sd_petall=sd(iris_data$Petal.Length)
sd_petalw=sd(iris_data$Petal.Width)

#coloured boxplots

plot1 <- ggplot(iris_data, aes(iris_data$Species, iris_data$Sepal.Length))
plot1 + geom_boxplot(aes(fill = iris_data$Species))

plot2 <- ggplot(iris_data, aes(iris_data$Species, iris_data$Sepal.Width))
plot2 + geom_boxplot(aes(fill = iris_data$Species))

plot3 <- ggplot(iris_data, aes(iris_data$Species, iris_data$Petal.Length))
plot3 + geom_boxplot(aes(fill = iris_data$Species))

plot4 <- ggplot(iris_data, aes(iris_data$Species, iris_data$Petal.Width))
plot4 + geom_boxplot(aes(fill = iris_data$Species))


