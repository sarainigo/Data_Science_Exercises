set.seed(123)
col_names <- c("mpg","cylinders","displacement","horsepower","weight","acceleration","model year","origin","car name")
data <- read.csv(url("https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"), header=FALSE, col.names = col_names, sep="", as.is=TRUE)
auto_data = data.frame(data)

horsepower <- as.numeric(auto_data$horsepower)
horsepower_median = median(horsepower, na.rm=TRUE)
horsepower_mean_1 = mean(horsepower, na.rm=TRUE)

num_NA = 0
for(i in 1:length(horsepower)) {
  if(is.na(horsepower[i])){
    horsepower[i]=horsepower_median
    num_NA = num_NA+1
    }
}

horsepower_mean_2 = mean(horsepower)
