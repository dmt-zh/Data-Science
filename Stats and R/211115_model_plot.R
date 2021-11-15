# Визуализируйте взаимодействие переменных wt и am, дополнив код, приведённый в задании:
#   Ось x - переменная wt
#   Ось y - переменная mpg
#   Цвет регрессионных прямых - переменная am


mtcars$am <- factor(mtcars$am, labels = c('Automatic', 'Manual'))
model <- lm(mpg ~ wt + am + wt * am, mtcars)
summary(model)

ggplot(mtcars, aes(x = wt, y = mpg, col = am)) + 
  geom_smooth(method = 'lm')
