# Постройте ящики с усами по каждой из характеристик (SAT ... Grad%) для каждого типа институтов.
# Какие из следующих утверждений верны (речь идет о наблюдаемых данных)?


df <- read.table('colleges.txt', sep = '\t', header = T)
df$School_Type <- factor(df$School_Type)
sub <- subset(df, School_Type == 'Univ')

library(ggplot2)
library(gridExtra)

g1 <- ggplot(df, aes(x=School_Type, y=SAT)) + 
  geom_boxplot() + facet_grid

g2 <- ggplot(df, aes(x=School_Type, y=Acceptance)) + 
  geom_boxplot()

g3 <- ggplot(df, aes(x=School_Type, y=X..Student)) + 
  geom_boxplot()

g4 <- ggplot(sub, aes(x=School, y=X..Student)) + 
  geom_boxplot()+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

grid.arrange(g1, g2, g3, g4, ncol = 4)


# % процент принятых студентов в школах искусств и в университетах примерно равны
# В одном из университетов тратят сильно больше денег в среднем на одного студента, чем в других университетах
# Разброс баллов SAT в университетах выше, чем в школах искусств
# Разброс средств, расходуемых в среднем на одного студента, в школах искусств меньше, чем в университетах







