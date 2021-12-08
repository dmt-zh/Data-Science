# Ќапишите функцию, котора€ принимает на вход два набора данных. ѕервый dataframe, как и в предыдущей задаче, содержит информацию 
# об уже осмотренном багаже (запрещенный или нет, вес, длина, ширина, тип сумки).

# ¬торой набор данных Ч это информаци€ о новом багаже, который сканируетс€ пр€мо сейчас. ¬ данных также есть информаци€:  вес, 
# длина, ширина, тип сумки и им€ пассажира (смотри описание переменных в примере). »спользу€ первый набор данных, обучите 
# регрессионную модель различать запрещенный и разрешенный багаж. ѕри помощи полученной модели дл€ каждого наблюдени€ в новых
# данных предскажите веро€тность того, что багаж €вл€етс€ запрещенным. ѕассажиров, чей багаж получил максимальное значение 
# веро€тности, мы попросим пройти дополнительную проверку. 

# »того, ваша функци€ принимает два набора данных и возвращает им€ пассажира с наиболее подозрительным багажом. ≈сли несколько
# пассажиров получили максимальное значение веро€тности, то верните вектор с несколькими именами. 

# ¬ этой задаче дл€ предсказани€ будем использовать все предикторы, даже если некоторые из них оказались незначимыми.
# ƒл€ предсказани€ стройте модель без взаимодействи€ предикторов.

# ѕример работы функции:

# > test_data <- read.csv("https://stepic.org/media/attachments/course/524/test_data_passangers.csv")
# > str(test_data)
# 'data.frame':  30 obs. of  5 variables:
# $ is_prohibited: Factor w/ 2 levels "No","Yes": 1 1 1 1 1 1 1 1 1 1 ...
# $ weight       : int  81 72 79 89 87 91 74 76 74 84 ...
# $ length       : int  49 49 60 49 54 42 54 49 49 53 ...
# $ width        : int  13 25 22 24 13 25 17 22 12 26 ...
# $ type         : Factor w/ 2 levels "Bag","Suitcase": 2 2 2 2 2 2 2 2 2 2 ...

# > data_for_predict <-read.csv("https://stepic.org/media/attachments/course/524/predict_passangers.csv")
# > str(data_for_predict)
# 'data.frame':  10 obs. of  5 variables:
# $ weight    : int  81 80 76 87 80 70 95 72 73 76
# $ length    : int  56 47 54 59 59 53 54 42 45 49
# $ width     : int  24 18 20 19 19 21 19 22 23 18
# $ type      : Factor w/ 2 levels "Bag","Suitcase": 2 1 1 1 2 1 2 2 2 1
# $ passangers: Factor w/ 10 levels "Anatoliy","Bob",..: 2 1 3 6 9 8 10 5 4 7

# > most_suspicious(test_data, data_for_predict)
# [1] Svetozar # пассажира звали —ветозар!


most_suspicious <- function(test_data, data_for_predict){
  fit <- glm(is_prohibited ~ ., test_data, family = "binomial")
  pred <- predict(fit, newdata = data_for_predict, type = "response")
  idx <- which(pred == max(pred))
  return(as.character(data_for_predict$passangers[idx]))
}

