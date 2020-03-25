set.seed(123)

# 5 number summary

trees_data = data.frame(trees)
summary(trees_data$Girth)
summary(trees_data$Height)
summary(trees_data$Volume)

fivenum(trees_data$Girth)
fivenum(trees_data$Height)
fivenum(trees_data$Volume)

# histograms
par(mfrow = c(3,1))
hist(trees_data$Girth)
hist(trees_data$Height)
hist(trees_data$Volume)

#skewness
sk_girth=skewness(trees_data$Girth)
sk_girth
sk_height=skewness(trees_data$Height)
sk_height
sk_volume=skewness(trees_data$Volume)
sk_volume

