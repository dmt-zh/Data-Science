# �������� ������� normality_test, ������� �������� �� ���� dataframe � ������������ ����������� ���������� ������ �����
# (��������������, ������, �������) � ��������� ������������ ������������� �������������� ����������. ������� ������ ����������
# ������ �������� p-������� ���������� ����� shapiro.test ��� ������ �������������� ����������.

# > normality_test(iris)
# Sepal.Length  Sepal.Width Petal.Length  Petal.Width 
# 1.018116e-02 1.011543e-01 7.412263e-10 1.680465e-08 

normality_test <- function(dataset){
  df <- dataset[sapply(dataset, function(x) is.numeric(x))]
  sapply(df, function(x) shapiro.test(x)$p.value)
}

