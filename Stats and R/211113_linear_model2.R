# ������������� ��� ��������� ������� diamonds �� ���������� ggplot2. ������ ��� ����������� ������ Ideal 
# (���������� cut) c ������ ����� ������ 0.46 (���������� carat) ��������� �������� ���������, ��� � ��������
# ��������� ���������� ��������� price, � �������� ���������� - ����������  depth. ��������� ������������ 
# ��������� � ���������� fit_coef.


special_diamonds <- subset(diamonds, cut == "Ideal" & carat == 0.46)
fit <- lm(price ~ depth, special_diamonds)
fit_coef <- fit$coefficients

# (Intercept)   depth 
# -76.11030    21.43427 

ggplot(special_diamonds, aes(depth, price))+
  geom_smooth()
