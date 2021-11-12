# В этой задаче вам дан набор данных, в котором представлена информация о температуре нескольких пациентов, 
# которые лечатся разными таблетками и у разных врачей.

# Проведите однофакторный дисперсионный анализ с повторными измерениями: влияние типа таблетки (pill) на 
# температуру (temperature) с учётом испытуемого (patient). Каково p-value для влияния типа таблеток на 
# температуру?
# Данные: 'Pillulkin.csv'



df <- read.csv('Pillulkin.csv')
str(df)

df$patient <- as.factor(df$patient)
df$pill <- as.factor(df$pill)
df$doctor <- as.factor(df$doctor)

ggplot(df, aes(y = temperature, x = pill, col = doctor)) + geom_boxplot()

res <- aov(temperature ~ pill + Error(patient/pill), data = df)
summary(res)

# Влияние таблеток на температуру является статистически незначимым, p-value > 0.05 (0.826)