# ���������� �������� ��� ������ mtcars. 
# ����� ��������� scatterplot � ������� ggplot �� ggplot2, �� ��� x �������� ����� mpg, �� ���
# y - disp, � ������ ���������� ���������� (hp).

# ���������� ������ ����� ��������� � ���������� plot1. ����� ������� � ������ ������ ���� ������:
#   
#   plot1 <- ggplot(data, aes())+
#     geom_****()


plot1 <- ggplot(mtcars, aes(x = mpg, y = disp, col = hp))+
  geom_point()