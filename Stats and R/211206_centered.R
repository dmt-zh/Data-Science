# ���� � ����� ������ ���� �������������� ����������, �� � ���������� �� ����� ����� ��������, ��������������� �������� ������
# �������������� ����������� � �������� ������ ��������������. ��� �� ������ ����������. ��������, ��� �� ��������� ������� ���
# ����� �������� �������� ��� �����. � ����� ��������� �������������� ���������� ����� ����� �������������� ������������ ���, 
# ����� ���� ������� ������� ��������� ����������. ����� ������� ������ ������������ ���������� � ������ �� ������� ���������� 
# ������� �������� ���� ����������.

# xcentered_i = x_i - mean(x)

# � ���� ������� ����� ������� �����  �������� ������� centered, ������� �������� �� ���� ��������� � ����� ����������, �������
# ���������� ������������ ���, ��� ��� ������� ����. ������� ������ ���������� ���� �� ���������, ������ � ��������������� 
# ���������� �����������.

# ������ ������ �������:
 
# test_data
# X1   X2   X3   X4
# 1  8.5  9.7 10.7 10.3
# 2  8.1 12.8  9.7 12.6
# 3  9.6  7.4  8.4 12.7
# 4  9.6 10.9  7.7  8.0
# 5 11.9 13.7 12.3 11.0 

# var_names = c("X4", "X2", "X1")
# centered(test_data, var_names)
# X1   X2   X3    X4
# 1 -1.04 -1.2 10.7 -0.62
# 2 -1.44  1.9  9.7  1.68
# 3  0.06 -3.5  8.4  1.78
# 4  0.06  0.0  7.7 -2.92
# 5  2.36  2.8 12.3  0.08



centered <- function(test_data, var_names){    
  test_data[var_names] <- sapply(test_data[var_names], function(x) x - mean(x))    
  return(test_data)    
}


