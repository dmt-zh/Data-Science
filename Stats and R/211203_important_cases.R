# �������� ����� ���������� important_cases - ������ � ����� ���������� ("No" � "Yes"). ���������� ������ ��������� 
# �������� Yes, ���� ��� ������� ������ �������� ���� �� ���� �������������� ���������� ���� ��������. � ��������� 
# ������ ���������� important_cases  ����� ��������� �������� No.
 
# ��������, ���������� ������ ������� ������ iris:
#   Sepal.Length Sepal.Width Petal.Length Petal.Width Species
# 1          5.1         3.5          1.4         0.2  setosa

# � ������ ������ ������ ��������  Sepal.Width 3.5 ������, ��� ������� �������� mean(Sepal.Width) = 3.057. 
# �������������� ��� ������� ������ �������� ���������� important_cases ����� "No".



vect_means <- sapply(iris[1:4], function(x) mean(x))
iris$important_cases <- factor(apply(iris[1:4], 1, function(x) if (sum(x > vect_means) >= 3) 'Yes' else 'No'))
