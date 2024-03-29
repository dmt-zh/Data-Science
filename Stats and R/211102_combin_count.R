# Пусть у нас есть n предметов, из которых нужно выбрать k штук.
# Известнейшая комбинаторная формула C_n^k = n!/k!(n-k)!задаёт количество всевозможных сочетаний.
# Похожий вид имеет и количество сочетаний с повторениями (мультикомбинаций). Запрограммируйте оба
# этих значения в виде функции, зависящей от n и k. Аргумент with_repetitions будет отвечать за 
# вариант подсчёта: если он FALSE, то пусть считается количество сочетаний, а если TRUE, то сочетаний 
# с повторениями.

combin_count <- function(n, k, with_repretitions = FALSE) {
  ifelse(with_repretitions == FALSE, choose(n, k), choose(k + n - 1, k))
}