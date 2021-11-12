# Проведите однофакторный дисперсионный анализ на встроенных данных iris. Зависимая переменная - ширина 
# чашелистика (Sepal.Width), независимая переменная - вид (Species). Затем проведите попарные сравнения 
# видов. Какие виды статистически значимо различаются по ширине чашелистика (p < 0.05)?

res <- aov(Sepal.Width ~ Species, data = iris)
summary(res)

TukeyHSD(res)

# Fit: aov(formula = Sepal.Width ~ Species, data = iris)

# Species
#                       diff          lwr        upr     p adj
# versicolor-setosa    -0.658 -0.81885528 -0.4971447 0.0000000
# virginica-setosa     -0.454 -0.61485528 -0.2931447 0.0000000
# virginica-versicolor  0.204  0.04314472  0.3648553 0.0087802

ggplot(iris, aes(x = Species, y = Sepal.Width)) + 
  geom_boxplot()