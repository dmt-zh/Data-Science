# �������� ������� get_difference, ������� �������� �� ���� ��� ���������: 
# test_data � ����� ������ � ������������ ������ �������������� ����������.
# n_cluster � ����� ���������, ������� ����� �������� � ������ ��� ������ ������������� �������������.
# ������� ������ ������� �������� ����������, �� ������� ���� ��������� �������� �������� ����� ����������� ���������� (p < 0.05).
# ����� �������, ����� ����, ��� �� �������� �������� ����� ���������, �� ��������� � �������� ������ ����� ������������ ���������� � �����
# ��������, � ���������� ������������ ������ ����� ����� �� �������������� ���������� ��� ������ �������������� �������.

# ������ ������ �������:
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


