# Постройте столбчатую диаграмму распределения цвета глаз по цвету волос только у женщин из
# таблицы HairEyeColor. По оси X должен идти цвет волос, цвет столбиков должен отражать цвет глаз.
# По оси Y - количество наблюдений. Чтобы построить столбчатую диаграмму в ggplot, вам нужно подключить
# нужный пакет, затем преобразовать таблицу HairEyeColor в data frame:
   
#   mydata <- as.data.frame(HairEyeColor)
 
# Постройте график на основе предложенного кода, сохранив его в переменную obj. Укажите, чему равен аргумент data,
#   что должно находиться в aes().
 
library("ggplot2")
mydata <- as.data.frame(HairEyeColor[, , 'Female'])
obj <- ggplot(data = mydata, aes(x = Hair, y = Freq, fill = Eye)) + 
  geom_bar(stat="identity", position=position_dodge()) + 
  scale_fill_manual(values=c("Brown", "Blue", "Darkgrey", "Darkgreen"))

