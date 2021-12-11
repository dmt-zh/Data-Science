# Напишите функцию smart_hclust, которая получает на вход dataframe  с произвольным числом количественных переменных
# и число кластеров, которое необходимо выделить при помощи иерархической кластеризации. Функция должна в исходный 
# набор данных добавлять новую переменную фактор - cluster  -- номер кластера, к которому отнесено каждое из наблюдений.
# 
# Пример работы функции:
#   
#   
#   > test_data <- read.csv("https://stepic.org/media/attachments/course/524/test_data_hclust.csv")
#   > str(test_data)
#   'data.frame':  12 obs. of  5 variables:
#   $ X1: int  11 9 9 9 7 9 16 23 15 19 ...
#   $ X2: int  7 10 2 11 9 11 20 18 21 20 ...
#   $ X3: int  10 10 12 8 10 9 22 21 14 15 ...
#   $ X4: int  10 8 14 10 11 6 19 24 21 17 ...
#   $ X5: int  8 6 11 3 14 9 16 16 21 17 ...
 
#   > smart_hclust(test_data, 3) # выделено три кластера
#   X1 X2 X3 X4 X5 cluster
#   1  11  7 10 10  8       1
#   2   9 10 10  8  6       1
#   3   9  2 12 14 11       1
#   4   9 11  8 10  3       1
#   5   7  9 10 11 14       1
#   6   9 11  9  6  9       1
#   7  16 20 22 19 16       2
#   8  23 18 21 24 16       2
#   9  15 21 14 21 21       3
#   10 19 20 15 17 17       3
#   11 20 24 21 20 19       2
#   12 22 19 27 22 19       2
  


smart_hclust <- function(test_data, cluster_number){
  dist_matrix <- dist(test_data)
  fit <- hclust(dist_matrix)
  cluster <- cutree(fit, cluster_number)
  test_data$cluster <- factor(cluster)
  return(test_data)
}

