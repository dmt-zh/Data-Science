# Постройте гистограммы числа покупателей 13го и 6го числа, а также гистограмму разницы числа посетителей. 
# Постройте ящики с усами числа покупателей 13го и 6го числа, а также их разницы.


df <- read.csv('13_6.csv', sep = '\t', header = F)
colnames(df) <- c('Data_type','Date','Fr6', 'Fr13', 'Supermarket')

df$Date_diff <- df$Fr13 - df$Fr6
n <- length(df$Fr13)
bins <- round(1 + 3.2*log(n))


boxplot(df[c(3, 4, 6)])

library(ggplot2)
library(gridExtra)

ggp1 <- ggplot(df, aes(x=Fr13))+
  geom_histogram(bins = bins, fill = 'blue', alpha=0.5)

ggp2 <- ggplot(df, aes(x=Fr6))+
  geom_histogram(bins = bins, fill = 'green', alpha=0.5)

ggp3 <- ggplot(df, aes(x=Date_diff))+
  geom_histogram(bins = bins, fill = 'red', alpha=0.5)

grid.arrange(ggp1, ggp2, ggp3, ncol = 3)


# Ответы:
#   На ящике с усами для разницы числа покупателей наблюдается выброс
#   "Усы" числа покупателей 6го и 13го числа практически не различаются
#   Гистограмма числа покупателей 6го числа имеет 2 пика
#   Форма гистограммы разницы числа покупателей куполообразная
#   Медиана разницы числа покупателей смещена к 1й квартили




