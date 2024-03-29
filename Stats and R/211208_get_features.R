# �����������, ��� �� �������� � ��������� � ������ ������������ � ��������� ����� ����������. � ����� ������������
# ���� ���������� � ����������� �������� ������ �� ���������� ������. ��� ������ ���� �� �����:
 
# ������� �� ����� ����������� - is_prohibited (No - �����������, Yes - �����������) 
# ��� ����� (��) - weight
# ����� (��) - length
# ������ (��) - width
# ��� ������ (����� ��� �������) - type.

# �������� ������� get_features , ������� �������� �� ���� ����� ������ � ������. ������ ������������� ���������, 
# ��� ��������� ���������� - ������� �� ����� �����������, � ���������� - ��������� ����������, � ���������� ������
# � ���������� ������������� �������� ���������� (p < 0.05) (� ������ ��� ��������������). ���� � ������ ��� �������� 
# �����������, ������� ���������� ������ � ����������  "Prediction makes no sense".

# > test_data <- read.csv("https://stepic.org/media/attachments/course/524/test_luggage_1.csv")
# > str(test_data)
# 'data.frame':	60 obs. of  5 variables:
# $ is_prohibited: Factor w/ 2 levels "No","Yes": 1 1 1 1 1 1 1 1 1 1 ...
# $ weight       : int  69 79 82 81 84 81 64 76 77 88 ...
# $ length       : int  53 52 54 50 48 51 53 52 53 52 ...
# $ width        : int  17 21 20 23 19 20 16 20 23 23 ...
# $ type         : Factor w/ 2 levels "Bag","Suitcase": 2 1 2 1 2 1 2 1 2 1 ...
# > get_features(test_data)
# [1] "Prediction makes no sense"


get_features <- function(dataset){
  fit <- glm(is_prohibited ~ ., dataset, family = "binomial")
  result <- anova(fit, test = "Chisq")
  features <- which(result$`Pr(>Chi)` < 0.05, arr.ind = T)
  if (length(features) > 0){
    return(row.names(result)[features])
  }else{
    return('Prediction makes no sense')
  }
}

