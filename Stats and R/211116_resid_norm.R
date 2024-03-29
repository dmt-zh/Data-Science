# �������� ������� resid.norm, ������� ��������� ������������� �������� �� ������ �� ������������ ��� ������ 
# ������� shapiro.test � ������� ����������� ��� ������ ������� ggplot() � ������� �������� "red", ���� 
# ������������� �������� ������� ���������� �� ����������� (p < 0.05), � � ������ �������� "green" - ���� 
# ������������� �������� ������� �� ���������� �� �����������.

# �� ���� ������� �������� ������������� ������. ������� ���������� ����������, � ������� �������� ������ ggplot.


resid.norm  <- function(fit){
  df <- as.data.frame(fit$residuals)
  color <- ifelse(shapiro.test(fit$residuals)$p > 0.05, 'green', 'red')
  plot <- ggplot(df, aes(x = fit$residuals)) +
           geom_histogram(fill = color)
  return(plot)
}

