# ������ ����� ������� ����� �������� ������������� ������������� ������ � ���������� �����������: ������� ��������
# doctor, ������� ������� pill � �� �������������� �� temperature. ������ ��� ��������������� ����������: � ��� ����,
# ��� ���� � ��� �� ������� ��������� ������ ��������, � ��� ����, ���  ���� � ��� �� ������� ������� � ������ ������!
# ������ F-�������� ��� �������������� �������� ������� (doctor) � ���� �������� (pill)?


df <- read.csv('Pillulkin.csv')
str(df)

df$patient <- as.factor(df$patient)
df$pill <- as.factor(df$pill)
df$doctor <- as.factor(df$doctor)

res <- aov(temperature ~ pill*doctor + Error(patient/(pill*doctor)), data = df)
summary(res)

# ������� �������� ������� (doctor) � ���� �������� (pill) �� ��������� ����������� � ������
# ���������� �� �������� ������������� �������� �-value > 0.05 (0.711).


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

