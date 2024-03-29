# �������� ������� get_coefficients, ������� �������� �� ���� dataframe � ����� ����������� x ( ������ � ������������ ������ ��������)
# � y ( ������ � ����� ����������). ������� ������ ������������� ������, ��� y � ��������� ����������, � x � �����������, � ����������
# ������ �� ��������� ���������� ������������� ������. 

# ������ ������ �������.
# test_data <- transform(test_data, x = factor(x), y = factor(y)) 
# get_coefficients(test_data) (Intercept)   x2    x3   0.9000000 2.5396825 0.6666667 


get_coefficients <- function(df){
  fit <- glm(y ~ x, df, family = "binomial")
  return(exp(coef(fit)))
}

