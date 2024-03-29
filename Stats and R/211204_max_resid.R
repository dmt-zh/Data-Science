# ������ ��������� ����� ����������� �����������, ����� �������� ������������� ���� ��������! ������� ������� ��� � ������� �������
# max_resid, ������� �������� �� ���� dataframe � ����� �����������: ����� ��������� � ����������� ��� ����������. 

# Drugs - ������ � ����� ����������: drug_1, drug_2, drug_3.     
# Result - ������ � ����� ����������: positive, negative.

# ������� ������ �������� ������ ������� ������������� � ������������  ��������� �������������������� ������� � ���������� ������ ��
# ���� ���������: �������� ������� � ������� ���� ������. ��� ������� ������������������� �������� �� ������ ��������������� ��� ��������
# ��� �������� chisq.test(). ������� ������� �� ���� �������, ����� �����, ��� �������� ������������������� �������.

# ������ ������ ������� �� ����� �� ���������:
# > test_data <- read.csv("https://stepic.org/media/attachments/course/524/test_drugs.csv")
# > str(test_data)
# 'data.frame':  395 obs. of  2 variables:
#   $ Drugs : Factor w/ 3 levels "drug_1","drug_2",..: 3 1 1 2 1 1 3 1 2 3 ...
#   $ Result: Factor w/ 2 levels "negative","positive": 2 1 1 2 1 2 2 2 1 1 ...

# > max_resid(test_data)
# [1] "drug_1"   "positive"


max_resid <- function(x){
  t <- table(x)
  fit <- chisq.test(t)$stdres
  ind <- which(fit == max(fit), arr.ind = T)
  return(c(rownames(ind), colnames(t)[ind][2]))
}

