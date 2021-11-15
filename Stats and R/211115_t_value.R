# ¬оспользуйтесь встроенным датасетом attitude, чтобы предсказать рейтинг (rating) по переменным complaints и critical. 
#  аково t-значение дл€ взаимодействи€ двух факторов?
  

model <- lm(rating ~ complaints * critical, attitude)
summary(model)$coefficients

# t value - 0.3163015
