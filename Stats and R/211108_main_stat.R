# Воспользуемся данными diamonds из библиотеки ggplot2. При помощи критерия Хи - квадрат проверьте гипотезу о 
# взаимосвязи качества огранки бриллианта (сut) и его цвета (color). В переменную main_stat сохраните значение
# статистики критерия Хи - квадрат. Обратите внимание, main_stat должен быть вектором из одного элемента, а не
# списком (листом).

library(ggplot2)
d_table <- table(diamonds$cut, diamonds$color)
main_stat <- chisq.test(d_table)$statistic


# Pearson's Chi-squared test
# data:  d_table
# X-squared = 310.32, df = 24, p-value < 2.2e-16

# Нет оснований для отклонения нулевой гипотезы
