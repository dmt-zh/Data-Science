# �������������� �������������� ���������� wt � am, �������� ���, ���������� � �������:
#   ��� x - ���������� wt
#   ��� y - ���������� mpg
#   ���� ������������� ������ - ���������� am


mtcars$am <- factor(mtcars$am, labels = c('Automatic', 'Manual'))
model <- lm(mpg ~ wt + am + wt * am, mtcars)
summary(model)

ggplot(mtcars, aes(x = wt, y = mpg, col = am)) + 
  geom_smooth(method = 'lm')
