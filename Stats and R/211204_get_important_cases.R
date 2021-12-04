# Ќапишем функцию get_important_cases, котора€ принимает на вход dataframe с произвольным числом количественных переменных (гарантируетс€ хот€ бы две переменные). 
# ‘ункци€ должна возвращать dataframe с новой переменной - фактором important_cases.
# ѕеременна€  important_cases принимает значение Yes, если дл€ данного наблюдени€ больше половины количественных переменных имеют значени€ больше среднего.
# ¬ противном случае переменна€ important_cases принимает значение No. ѕеременна€  important_cases - фактор с двум€ уровн€ми 0 - "No", 1  - "Yes".  
# “о есть даже если в каком-то из тестов все наблюдени€ получили значени€ "No", фактор должен иметь две градации. 


get_important_cases <- function(df){
  n <- ncol(df)
  vect_means <- sapply(df[1:n], function(x) mean(x))
  df$important_cases <- factor(apply(df[1:n], 1, function(x) if (sum(x > vect_means) > n/2) 'Yes' else 'No'),
                               levels = c('No', 'Yes'))
  
  return(df) 
}



