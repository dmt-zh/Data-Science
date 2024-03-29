# ������� ������� get_important_cases, ������� ��������� �� ���� dataframe � ������������ ������ �������������� ���������� (������������� ���� �� ��� ����������). 
# ������� ������ ���������� dataframe � ����� ���������� - �������� important_cases.
# ����������  important_cases ��������� �������� Yes, ���� ��� ������� ���������� ������ �������� �������������� ���������� ����� �������� ������ ��������.
# � ��������� ������ ���������� important_cases ��������� �������� No. ����������  important_cases - ������ � ����� �������� 0 - "No", 1  - "Yes".  
# �� ���� ���� ���� � �����-�� �� ������ ��� ���������� �������� �������� "No", ������ ������ ����� ��� ��������. 


get_important_cases <- function(df){
  n <- ncol(df)
  vect_means <- sapply(df[1:n], function(x) mean(x))
  df$important_cases <- factor(apply(df[1:n], 1, function(x) if (sum(x > vect_means) > n/2) 'Yes' else 'No'),
                               levels = c('No', 'Yes'))
  
  return(df) 
}



