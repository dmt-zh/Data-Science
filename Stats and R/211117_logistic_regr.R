# ���������� ��������� ������ � ����������� �������� � ������� ����������� � ������������ ������. ������ �������� �� 
# ������: https://stepic.org/media/attachments/lesson/11478/data.csv ��� ����� ���������� ��������, ��������� ��� � 
# ����������� ��� ��� (���������� admit, 1 = ���������, 0 = �� ���������), ��� ��������� ����� ������ ��� (NA). 
# �������� ������ (�������� �� ���� �������� ��� ���������� ���������):
   
# 'data.frame':  400 obs. of  4 variables:
# $ admit: Factor w/ 2 levels "0","1": 1 2 NA NA 1 2 NA NA 2 1 ...
# $ gre  : int  380 660 800 640 520 760 560 400 540 700 ...
# $ gpa  : num  3.61 3.67 4 3.19 2.93 3 2.98 3.08 3.39 3.92 ...
# $ rank : Factor w/ 4 levels "1","2","3","4": 3 3 1 4 4 2 1 2 3 2 ...

# �� ��������� ������ � ���������� admit ��������� ������������� ������������� ������, ��������������� ��������� 
# ����������� �� ������������ �������� ��������� �������� ����������� (���������� rank, 1 � �������� ����������, 
# 4 � �������� ����������) � ����������� GPA (���������� gpa) � ������ �� ��������������. ��������� ��� ������ � 
# ��� ����� ������, ��� ��������� ����������� ����������.

# ������� � ������ ����� ������������� ������� ����� ����������� �� ���, ��� ���� ��������� ����������� ��� ����������. 
# ������� �������� �����������, ����� ����������� ��� ����������� �� ������ 0.4.


# ��������� ������ � �������� ���������� admit � rank �� ������
df <- read.csv('https://stepic.org/media/attachments/lesson/11478/data.csv')
df$admit <- factor(dta1$admit, labels = c("N", "Y"))
df$rank <- factor(dta1$rank, labels = c("A","B","C","D"))

# ������� ������� ������ ��� NA
train <- na.omit(df)

# ������� ������� ������ � NA ��� ������������
test <- subset(df, is.na(df$admit) == T)

# ������ �� train ������� ������ ������������� ���������
model <- glm(admit ~ rank * gpa, data=train, family = 'binomial')

# �������� ��������� ������ ��� test ������
test$prop <- predict(model, newdata = test, type = "response")

# ��������� ���������� �����������
sum(test$prop >= 0.4)
# 56






