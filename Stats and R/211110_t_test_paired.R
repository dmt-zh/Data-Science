# �������� ������, ����������� ������� ���������� ���� ������� �� ���������� ������������� ��������. 
# https://stepic.org/media/attachments/lesson/11504/lekarstva.csv???
# �� ���� ���������� �������� ���������� �������� �� ������ ������� (Pressure_before) � ����������� 
# �������� ����� ������� (Pressure_after) ��� ������ t - �������� ��� ��������� �������. 
# � ���� ��� ������ ������� �������� t - ��������.


med <- read.csv('lekarstva.csv')
str(med)

med$Group <- as.factor(med$Group)
t.test(med$Pressure_before, med$Pressure_after, paired = T)$statistic
