# В данных swiss, используя все переменные, выделите два кластера при помощи иерархической кластеризации
# и сохраните значение кластеров как фактор в переменную cluster.
# Затем визуализируйте взаимосвязь переменных  Education и  Catholic в двух выделенных кластерах.


smart_hclust <- function(test_data, cluster_number){
  dist_matrix <- dist(test_data)
  fit <- hclust(dist_matrix)
  cluster <- cutree(fit, cluster_number)
  test_data$cluster <- factor(cluster)
  return(test_data)
}


df <- smart_hclust(swiss, 2)

library(ggplot2)
ggplot(df, aes(Education, Catholic, col = cluster)) +
  geom_point() + 
  geom_smooth(method = 'lm')