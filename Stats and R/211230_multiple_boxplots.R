# ��������� ����� � ����� �� ������ �� ������������� (SAT ... Grad%) ��� ������� ���� ����������.
# ����� �� ��������� ����������� ����� (���� ���� � ����������� ������)?


df <- read.table('colleges.txt', sep = '\t', header = T)
df$School_Type <- factor(df$School_Type)
sub <- subset(df, School_Type == 'Univ')

library(ggplot2)
library(gridExtra)

g1 <- ggplot(df, aes(x=School_Type, y=SAT)) + 
  geom_boxplot() + facet_grid

g2 <- ggplot(df, aes(x=School_Type, y=Acceptance)) + 
  geom_boxplot()

g3 <- ggplot(df, aes(x=School_Type, y=X..Student)) + 
  geom_boxplot()

g4 <- ggplot(sub, aes(x=School, y=X..Student)) + 
  geom_boxplot()+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

grid.arrange(g1, g2, g3, g4, ncol = 4)


# % ������� �������� ��������� � ������ �������� � � ������������� �������� �����
# � ����� �� ������������� ������ ������ ������ ����� � ������� �� ������ ��������, ��� � ������ �������������
# ������� ������ SAT � ������������� ����, ��� � ������ ��������
# ������� �������, ����������� � ������� �� ������ ��������, � ������ �������� ������, ��� � �������������







