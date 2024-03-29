# �������� ������� is_multicol, ������� �������� �� ���� dataframe ������������� ������� � ��������������� �����������. 
# ������� ������ ��������� ������������� ������� ��������������������, � ������ ������� �������� ���������� ����� ������������.
# �������� ����������� �������� ��������, ����� ���� ���������� ����� ���� �������� ����� ������ ���������� ��� ������ ���������
# �������� V1 = V2 + 4 ��� V1 = V2 - 5.
# ������� ���������� ����� ����������, ����� �������� ���� �������� ����������� ��� c������� "There is no collinearity in the data".

# ������ ������ �������:
  
# test_data <- read.csv("https://stepic.org/media/attachments/course/524/Norris_1.csv")
# V1 V2 V3 V4
# 1 22 20 18 20
# 2 16 28 31 15
# 3 14 24  7 16
# is_multicol(test_data)
# "There is no collinearity in the data"
  

# test_data <- read.csv("https://stepic.org/media/attachments/course/524/Norris_2.csv")
# V1 V2 V3 V4
# 1 13 12  7 11
# 2 15 14 13 10
# 3  8  7 11 16
# is_multicol(test_data)
# [1] "V2" "V1"
  


is_multicol <- function(data){
  fit <- cor(data)
  diag(fit) <- 0
  vect <- rownames(which(abs(round(fit, 3)) == 1, arr.ind = T))
  if(length(vect) > 0){
    return(vect)
  }else{
    return('There is no collinearity in the data')
  }
}









