# �������������� ���������� ��������� attitude, ����� ����������� ������� (rating) �� ���������� complaints � critical. 
# ������ t-�������� ��� �������������� ���� ��������?
  

model <- lm(rating ~ complaints * critical, attitude)
summary(model)$coefficients

# t value - 0.3163015
