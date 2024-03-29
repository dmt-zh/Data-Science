# �������� ������� smart_anova, ������� �������� �� ���� dataframe � ����� ����������� x � y. ���������� x � ��� ��������������
# ����������, ���������� y - ������, ��������� ���������� �� ��� ������.
# ���� ������������� �� ���� ������� ������� �� ���������� �� �����������, � ��������� � ������� ���������, ������� ������ ��������
# ��� ������ ��� ������ �������������� ������� � ������� ???����������� ������ �� ��������� p-value, ��� �������� � "ANOVA".
# 
# ���� ���� �� � ����� ������ ������������� ������� ���������� �� ����������� ��� ��������� �����������, ������� ���������� ������
# ��� ������ �������� �������� � ������� � ���������� ����������� ������ �� ��������� p-value, ��� �������  � "KW".
# ������������� ����� ������� ������� ������������� �� �����������, ���� � ����� shapiro.test() p < 0.05.
# ��������� ����� ������� �� �����������, ���� � ����� bartlett.test() p < 0.05.
# 
# ������ ������ �������:
#   > test_data <- read.csv("https://stepic.org/media/attachments/course/524/s_anova_test.csv")
#   > str(test_data)
#   'data.frame':	30 obs. of  2 variables:
#   $ x: num  1.08 0.07 -1.02 -0.45 0.81 -1.27 -0.75 1.47 -0.2 -1.48 ...
#   $ y: Factor w/ 3 levels "A","B","C": 1 1 1 1 1 1 1 1 1 1 ...
#   > smart_anova(test_data)
#   ANOVA 
#   0.265298
  

smart_anova <- function(test_data){
  p_sh <- aggregate(x ~ y, test_data, function(x) shapiro.test(x)$p.value)$x
  p_bt <- bartlett.test(x ~ y, test_data)$p.value
  if (all(p_sh > 0.05) & p_bt > 0.05){
    fit <- aov(x ~ y, test_data)
    anv <- c(ANOVA = summary(fit)[[1]]$'Pr(>F)'[1])
    return(anv)
  } else {
    kw <- c(KW = kruskal.test(x ~ y, test_data)$p.)
    return(kw)
  }
}


