# Постройте гистограммы по каждой из характеристик (SAT ... Grad%) для каждого типа институтов. 
# Какие из следующих утверждений верны (речь идет о наблюдаемых данных)?


df <- read.table('colleges.txt', sep = '\t', header = T)
df$School_Type <- factor(df$School_Type)

n <- length(df$SAT)
bins <- round(1 + 3.2*log(n))

sub <- subset(df, School_Type == 'Univ')

library(ggplot2)
library(gridExtra)

g1 <- ggplot(df, aes(x=X.PhD, fill=School_Type))+
  geom_histogram(bins = bins)+
  facet_wrap(~School_Type)

g2 <- ggplot(df, aes(x=Grad., fill=School_Type))+
  geom_histogram(bins = bins)+
  facet_wrap(~School_Type)


grid.arrange(g1, g2, ncol = 2)


# Распределение процента факультетов с Phd в школах искусств напоминает "нормальное"
# В университетах процент факультетов с Phd выше
# Гистограмма процента студентов, которые успешно завершили университеты, имеет несколько пиков
