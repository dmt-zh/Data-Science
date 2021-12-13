# Напишите функцию get_pca2, которая принимает на вход dataframe с произвольным числом количественных переменных.
# Функция должна рассчитать, какое минимальное число главных компонент объясняет больше 90% изменчивости в исходных
# данных и добавлять значения этих компонент в исходный dataframe в виде новых переменных.
# 
# Рассмотрим работу функции на примере встроенных данных swiss:
#   
# result  <- get_pca2(swiss)
# str(result)
# 'data.frame':  47 obs. of  8 variables:
# $ Fertility       : num  80.2 83.1 92.5 85.8 76.9 76.1 83.8 92.4 82.4 82.9 ...
# $ Agriculture     : num  17 45.1 39.7 36.5 43.5 35.3 70.2 67.8 53.3 45.2 ...
# $ Examination     : int  15 6 5 12 17 9 16 14 12 16 ...
# $ Education       : int  12 9 5 7 15 7 7 8 7 13 ...
# $ Catholic        : num  9.96 84.84 93.4 33.77 5.16 ...
# $ Infant.Mortality: num  22.2 22.2 20.2 20.3 20.6 26.6 23.6 24.9 21 24.4 ...
# $ PC1             : num  37.03 -42.8 -51.08 7.72 35.03 ...
# $ PC2             : num  -17.43 -14.69 -19.27 -5.46 5.13 ...



get_pca2 <- function(data){
  fit <- prcomp(data)
  res <- as.vector(summary(fit)$importance[3,])
  vect <- length(res[res < 0.9]) + 1
  data <- cbind(data, fit$x[, 1:vect])
  return(data)
}

