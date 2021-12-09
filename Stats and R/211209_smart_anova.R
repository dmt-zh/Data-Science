# Напишите функцию smart_anova, которая получает на вход dataframe с двумя переменными x и y. Переменная x — это количественная
# переменная, переменная y - фактор, разбивает наблюдения на три группы.
# Если распределения во всех группах значимо не отличаются от нормального, а дисперсии в группах гомогенны, функция должна сравнить
# три группы при помощи дисперсионного анализа и вернуть ???именованный вектор со значением p-value, имя элемента — "ANOVA".
# 
# Если хотя бы в одной группе распределение значимо отличается от нормального или дисперсии негомогенны, функция сравнивает группы
# при помощи критерия Краскела — Уоллиса и возвращает именованный вектор со значением p-value, имя вектора  — "KW".
# Распределение будем считать значимо отклонившимся от нормального, если в тесте shapiro.test() p < 0.05.
# Дисперсии будем считать не гомогенными, если в тесте bartlett.test() p < 0.05.
# 
# Пример работы функции:
#   > test_data <- read.csv("https://stepic.org/media/attachments/course/524/s_anova_test.csv")
#   > str(test_data)
#   'data.frame':	30 obs. of  2 variables:
#   $ x: num  1.08 0.07 -1.02 -0.45 0.81 -1.27 -0.75 1.47 -0.2 -1.48 ...
#   $ y: Factor w/ 3 levels "A","B","C": 1 1 1 1 1 1 1 1 1 1 ...
#   > smart_anova(test_data)
#   ANOVA 
#   0.265298
  

smart_anova <- function(test_data){
  p_sh <- aggregate(x ~ y, test_data, function(x) shapiro.test(x)$p.value)$x
  p_bt <- bartlett.test(x ~ y, test_data)$p.value
  if (all(p_sh > 0.05) & p_bt > 0.05){
    fit <- aov(x ~ y, test_data)
    anv <- c(ANOVA = summary(fit)[[1]]$'Pr(>F)'[1])
    return(anv)
  } else {
    kw <- c(KW = kruskal.test(x ~ y, test_data)$p.)
    return(kw)
  }
}


