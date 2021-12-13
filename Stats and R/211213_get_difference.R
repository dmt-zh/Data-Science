# Ќапишите функцию get_difference, котора€ получает на вход два аргумента: 
# test_data Ч набор данных с произвольным числом количественных переменных.
# n_cluster Ч число кластеров, которое нужно выделить в данных при помощи иерархической кластеризации.
# ‘ункци€ должна вернуть названи€ переменных, по которым были обнаружен значимые различи€ между выделенными кластерами (p < 0.05).
# »ными словами, после того, как мы выделили заданное число кластеров, мы добавл€ем в исходные данные новую группирующую переменную Ч номер
# кластера, и сравниваем получившиес€ группы между собой по количественным переменным при помощи дисперсионного анализа.

# ѕример работы функции:
# test_data <- read.csv("https://stepic.org/media/attachments/course/524/cluster_2.csv")
# get_difference(test_data, 2)
# [1] "V1" "V2"


get_difference <- function(df, cluster_number){
  dist_matrix <- dist(df)
  fit <- hclust(dist_matrix)
  cluster <- cutree(fit, cluster_number)
  df$cluster <- factor(cluster)
  res <- sapply(df[, -ncol(df)], function(x) summary(aov(x ~ cluster, data = df))[[1]]$'Pr(>F)'[1])
  return(names(res[res < 0.05]))
  
}


