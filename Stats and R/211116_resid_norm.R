# Напишите функцию resid.norm, которая тестирует распределение остатков от модели на нормальность при помощи 
# функции shapiro.test и создает гистограмму при помощи функции ggplot() с красной заливкой "red", если 
# распределение остатков значимо отличается от нормального (p < 0.05), и с зелёной заливкой "green" - если 
# распределение остатков значимо не отличается от нормального.

# На вход функция получает регрессионную модель. Функция возвращает переменную, в которой сохранен график ggplot.


resid.norm  <- function(fit){
  df <- as.data.frame(fit$residuals)
  color <- ifelse(shapiro.test(fit$residuals)$p > 0.05, 'green', 'red')
  plot <- ggplot(df, aes(x = fit$residuals)) +
           geom_histogram(fill = color)
  return(plot)
}

