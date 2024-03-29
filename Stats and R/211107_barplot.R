# ��������� ���������� ��������� ������������� ����� ���� �� ����� ����� ������ � ������ ��
# ������� HairEyeColor. �� ��� X ������ ���� ���� �����, ���� ��������� ������ �������� ���� ����.
# �� ��� Y - ���������� ����������. ����� ��������� ���������� ��������� � ggplot, ��� ����� ����������
# ������ �����, ����� ������������� ������� HairEyeColor � data frame:
   
#   mydata <- as.data.frame(HairEyeColor)
 
# ��������� ������ �� ������ ������������� ����, �������� ��� � ���������� obj. �������, ���� ����� �������� data,
#   ��� ������ ���������� � aes().
 
library("ggplot2")
mydata <- as.data.frame(HairEyeColor[, , 'Female'])
obj <- ggplot(data = mydata, aes(x = Hair, y = Freq, fill = Eye)) + 
  geom_bar(stat="identity", position=position_dodge()) + 
  scale_fill_manual(values=c("Brown", "Blue", "Darkgrey", "Darkgreen"))

