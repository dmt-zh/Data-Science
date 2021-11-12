# Теперь вашей задачей будет провести двухфакторный дисперсионный анализ с повторными измерениями: влияние факторов
# doctor, влияние фактора pill и их взаимодействие на temperature. Учтите обе внутригрупповые переменные: и тот факт,
# что один и тот же больной принимает разные таблетки, и тот факт, что  один и тот же больной лечится у разных врачей!
# Каково F-значение для взаимодействия факторов доктора (doctor) и типа таблеток (pill)?


df <- read.csv('Pillulkin.csv')
str(df)

df$patient <- as.factor(df$patient)
df$pill <- as.factor(df$pill)
df$doctor <- as.factor(df$doctor)

res <- aov(temperature ~ pill*doctor + Error(patient/(pill*doctor)), data = df)
summary(res)

# Влияние факторов доктора (doctor) и типа таблеток (pill) на изменение температуры у разных
# испытуемых НЕ является статистически значимым р-value > 0.05 (0.711).


# Error: patient
# Df Sum Sq Mean Sq F value Pr(>F)
# Residuals  9  42.82   4.758               
# 
# Error: patient:pill
# Df Sum Sq Mean Sq F value Pr(>F)
# pill       1  0.133   0.133   0.051  0.826
# Residuals  9 23.479   2.609               
# 
# Error: patient:doctor
# Df Sum Sq Mean Sq F value Pr(>F)
# doctor     1  15.70  15.696   3.113  0.111
# Residuals  9  45.37   5.042               
# 
# Error: patient:pill:doctor
# Df Sum Sq Mean Sq F value Pr(>F)
# pill:doctor  1  0.422  0.4215   0.146  0.711
# Residuals    9 26.014  2.8905

