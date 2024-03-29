# �������� ������� normality_by, ������� ��������� �� ���� dataframe c ����� �����������. ������ ���������� ��������������, 
# ������ � ������ ����� ��� �������� � ��������� ���� ���������� �� ������. ������� ������ ��������� ������������� �� ������������
# � ������ ������������ ������ � ���������� dataframe � ������������ ���������� ����� shapiro.test

# ������ ������ �������:
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

