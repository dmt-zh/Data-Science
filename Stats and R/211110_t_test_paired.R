# Скачайте данные, посвященные влиянию различного типа лечения на показатель артериального давления. 
# https://stepic.org/media/attachments/lesson/11504/lekarstva.csv???
# По всем испытуемым сравните показатель давления до начала лечения (Pressure_before) с показателем 
# давления после лечения (Pressure_after) при помощи t - критерия для зависимых выборок. 
# В поле для ответа укажите значение t - критерия.


med <- read.csv('lekarstva.csv')
str(med)

med$Group <- as.factor(med$Group)
t.test(med$Pressure_before, med$Pressure_after, paired = T)$statistic
