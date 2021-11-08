# На основе таблицы HairEyeColor создайте ещё одну таблицу, в которой хранится информация о распределении цвета 
# глаз у женщин-шатенок (Hair = 'Brown'). Проведите тест равномерности распределения цвета глаз у шатенок и 
# выведите значение хи-квадрата для этого теста.


chisq.test(HairEyeColor['Brown', , 'Female'])

# Chi-squared test for given probabilities
# data:  HairEyeColor["Brown", , "Female"]
# X-squared = 40.189, df = 3, p-value = 9.717e-09
