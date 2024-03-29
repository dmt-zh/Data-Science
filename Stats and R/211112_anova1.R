# � ���� ������ ��� ��� ����� ������, � ������� ������������ ���������� � ����������� ���������� ���������, 
# ������� ������� ������� ���������� � � ������ ������.

# ��������� ������������� ������������� ������ � ���������� �����������: ������� ���� �������� (pill) �� 
# ����������� (temperature) � ������ ����������� (patient). ������ p-value ��� ������� ���� �������� �� 
# �����������?
# ������: 'Pillulkin.csv'



df <- read.csv('Pillulkin.csv')
str(df)

df$patient <- as.factor(df$patient)
df$pill <- as.factor(df$pill)
df$doctor <- as.factor(df$doctor)

ggplot(df, aes(y = temperature, x = pill, col = doctor)) + geom_boxplot()

res <- aov(temperature ~ pill + Error(patient/pill), data = df)
summary(res)

# ������� �������� �� ����������� �������� ������������� ����������, p-value > 0.05 (0.826)