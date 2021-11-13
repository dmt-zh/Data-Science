# ¬оспользуемс€ уже знакомыми данными diamonds из библиотеки ggplot2. “олько дл€ бриллиантов класса Ideal 
# (переменна€ cut) c числом карат равным 0.46 (переменна€ carat) постройте линейную регрессию, где в качестве
# зависимой переменной выступает price, в качестве предиктора - переменна€  depth. —охраните коэффициенты 
# регрессии в переменную fit_coef.


special_diamonds <- subset(diamonds, cut == "Ideal" & carat == 0.46)
fit <- lm(price ~ depth, special_diamonds)
fit_coef <- fit$coefficients

# (Intercept)   depth 
# -76.11030    21.43427 

ggplot(special_diamonds, aes(depth, price))+
  geom_smooth()
