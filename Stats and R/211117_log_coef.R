# ���������� ������ mtcars. ��������� � ���������� ������������� ������������� ������, ��� � ��������
# ��������� ���������� ��������� ��� ������� ������� (am), � �������� ����������� ���������� disp, vs, mpg.
# �������� ������������� ��������� ��������� � ���������� log_coef.


fit  <- glm(am ~ disp + vs + mpg, mtcars, family = binomial())
log_coef <- fit$coefficients
