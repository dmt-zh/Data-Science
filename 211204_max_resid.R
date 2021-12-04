# Доктор Пилюлькин решил вооружиться статистикой, чтобы сравнить эффективность трех лекарств! Давайте поможем ему и напишем функцию
# max_resid, которая получает на вход dataframe с двумя переменными: типом лекарства и результатом его применения. 

# Drugs - фактор с тремя градациями: drug_1, drug_2, drug_3.     
# Result - фактор с двумя градациями: positive, negative.

# Функция должна находить ячейку таблицы сопряженности с максимальным  значением стандартизированного остатка и возвращать вектор из
# двух элементов: название строчки и столбца этой ячейки. Для расчета стандартизированных остатков вы можете воспользоваться уже знакомой
# вам функцией chisq.test(). Изучите справку по этой функции, чтобы найти, где хранятся стандартизированные остатки.

# Пример работы функции на одном из вариантов:
# > test_data <- read.csv("https://stepic.org/media/attachments/course/524/test_drugs.csv")
# > str(test_data)
# 'data.frame':  395 obs. of  2 variables:
#   $ Drugs : Factor w/ 3 levels "drug_1","drug_2",..: 3 1 1 2 1 1 3 1 2 3 ...
#   $ Result: Factor w/ 2 levels "negative","positive": 2 1 1 2 1 2 2 2 1 1 ...

# > max_resid(test_data)
# [1] "drug_1"   "positive"


max_resid <- function(x){
  t <- table(x)
  fit <- chisq.test(t)$stdres
  ind <- which(fit == max(fit), arr.ind = T)
  return(c(rownames(ind), colnames(t)[ind][2]))
}

