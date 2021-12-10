# Напишите функцию normality_by, которая принимает на вход dataframe c тремя переменными. Первая переменная количественная, 
# вторая и третья имеют две градации и разбивают наши наблюдения на группы. Функция должна проверять распределение на нормальность
# в каждой получившейся группе и возвращать dataframe с результатами применения теста shapiro.test

# Пример работы функции:
# normality_by(mtcars[, c("mpg", "am", "vs")])
# am vs   p_value
# 1  0  0 0.5041144
# 2  0  1 0.6181271
# 3  1  0 0.5841903
# 4  1  1 0.4168822


normality_by <- function(df){
  colnames(df)[1] <- 'p_value'
  table <- aggregate(p_value ~ ., df, function(x) shapiro.test(x)$p.value)
  return(table)
}

