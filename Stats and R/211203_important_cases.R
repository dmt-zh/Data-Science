# Создайте новую переменную important_cases - фактор с двумя градациями ("No" и "Yes"). Переменная должна принимать 
# значение Yes, если для данного цветка значения хотя бы трех количественных переменных выше среднего. В противном 
# случае переменная important_cases  будет принимать значение No.
 
# Например, рассмотрим первую строчку данных iris:
#   Sepal.Length Sepal.Width Petal.Length Petal.Width Species
# 1          5.1         3.5          1.4         0.2  setosa

# В данном случае только значение  Sepal.Width 3.5 больше, чем среднее значение mean(Sepal.Width) = 3.057. 
# Соответственно для первого цветка значение переменной important_cases будет "No".



vect_means <- sapply(iris[1:4], function(x) mean(x))
iris$important_cases <- factor(apply(iris[1:4], 1, function(x) if (sum(x > vect_means) >= 3) 'Yes' else 'No'))
