"Uma empresa da ?rea de transporte e log?stica deseja migrar sua frota para carros el?tricos com o objetivo de reduzir os custos.

Antes de tomar a decis?o, a empresa gostaria de prever o consumo de energia de carros el?tricos com base em diversos fatores de utiliza??o e caracter?sticas dos ve?culos.

Usando um incr?vel dataset com dados reais dispon?veis publicamente, voc? dever? construir um modelo de Machine Learning capaz de prever o consumo de energia de carros
el?tricos com base em diversos fatores, tais como o tipo e n?mero de motores el?tricos do ve?culo, o peso do ve?culo, a capacidade de carga, entre outros atributos.

Para a constru??o desse projeto, recomendamos a utiliza??o da linguagem R e o dataset dispon?vel para download no link abaixo:

https://data.mendeley.com/datasets/tb9yrptydn/2

Este conjunto de dados lista todos os carros totalmente el?tricos com seus atributos (propriedades) dispon?veis atualmente no mercado. A cole??o n?o cont?m dados sobre carros
h?bridos e carros el?tricos dos chamados "extensores de alcance". Os carros a hidrog?nio tamb?m n?o foram inclu?dos no conjunto de dados devido ao n?mero insuficiente de modelos
produzidos em massa e ? especificidade diferente (em compara??o com ve?culo el?trico) do ve?culo, incluindo os diferentes m?todos de carregamento.

O conjunto de dados inclui carros que, a partir de 2 de dezembro de 2020, poderiam ser adquiridos na Pol?nia como novos em um revendedor autorizado e aqueles dispon?veis em pr?-
venda p?blica e geral, mas apenas se uma lista de pre?os publicamente dispon?vel com vers?es de equipamentos e par?metros t?cnicos completos estivesse dispon?vel. A lista n?o inclui carros
descontinuados que n?o podem ser adquiridos como novos de um revendedor autorizado (tamb?m quando n?o est?o dispon?veis em estoque).

O conjunto de dados de carros el?tricos inclui todos os carros totalmente el?tricos no mercado prim?rio que foram obtidos de materiais oficiais (especifica??es t?cnicas e cat?logos)
fornecidos por fabricantes de autom?veis com licen?a para vender carros na Pol?nia. Esses materiais foram baixados de seus sites oficiais. Caso os dados fornecidos pelo fabricante
estivessem incompletos, as informa??es eram complementadas com dados do AutoCat?logo SAMAR (link dispon?vel na se??o Refer?ncias da fonte de dados).

Seu trabalho ? construir um modelo de Machine Learning capaz de prever o consumo de energia de ve?culos el?tricos."

#ATINGIR pelo menos 85% de acuracia.


setwd('C:/Users/Matheus Keitaro/OneDrive - Tecsoil Automa??o e Sistemas S.A/?rea de Trabalho/DSA/R e Azure Machine learning/Projetos-1-2/Projeto 1/tb9yrptydn-2')
getwd()

library(dplyr)
library(tidyr)
library(caret)
library(corrplot)
library(readxl)
library(caTools)
library(glmnet)
library(e1071)
library(rpart)


#Lendo o arquivo que contem o Dataset do projeto
df <- read_excel('FEV-data-Excel.xlsx')

#Visualizando os dados
class(df)
str(df)
summary(df)
dim(df)
View(df)

#Visualizando as marcas de carros o dataset possui
unique(df$Make)

#Verificando valores NA no dataset
colSums(is.na(df)) 

#Como os carros da tesla est?o com valores NA para variavel Consume kwh/100kmh, foi optado a ser dropada
df <- subset(df, select= -`mean - Energy consumption [kWh/100 km]`)

#De acordo com o site: https://www.evspecifications.com/en/model/bf3f173 o type of break da mercedez (NA) ? Front and rear brakes.
df$`Type of brakes` <- ifelse(df$`Car full name` == "Mercedes-Benz EQV (long)", 'disc (front + rear)', df$`Type of brakes`)

#Para as colunas com valores NA, esta sendo substituida pela m?dia, para ver a importancia na correla??o da determinada variavel com a variavel target.

df_num <- df %>% select(where(is.numeric))
df_num <- df_num %>% mutate_all(~ifelse(is.na(.x), mean(.x, na.rm = TRUE), .x))

'#normalizando os dados
max <- apply(df_num, 2, max)
min <- apply(df_num, 2, min)

df_num <- as.data.frame(scale(df_num, center = min, scale = max-min))'

#Para as colunas categoricas, esta sendo convertido do tipo char para Factor.
df_cat <- df %>% select(where(is.character))
df_cat[sapply(df_cat, is.character)] <- lapply(df_cat[sapply(df_cat, is.character)], 
                                       as.factor)

#Obtendo as vari?veis num?ricas

View(df_num)
corr <- cor(df_num)
corr

#Corrplot
corrplot(corr, method = "color")

#Verificando se a variavel target tem uma tendencia normal
shapiro.test(df_num$`Range (WLTP) [km]`)
hist(df$`Range (WLTP) [km]`)
#p-value > 0.05


#Splitando os dados
sample <- sample.split(df_num$`Range (WLTP) [km]`, SplitRatio = 0.7)

#Numerical dataset

df_train <- subset(df_num, sample == TRUE)
df_teste <- subset(df_num, sample == FALSE)

#Numerical + categorical dataset
set.seed(1234)
df_train2 <- subset(cbind(df_cat, df_num), sample == TRUE)
df_train2 <- subset(df_train2, select= -c(`Car full name`, Model))
set.seed(1234)
df_teste2 <- subset(cbind(df_cat, df_num), sample == FALSE)
df_teste2 <- subset(df_teste2, select= -c(`Car full name`, Model))

View(df_train2)
View(df_teste2)

#Treinando alguns modelos
modelov1 <- lm(`Range (WLTP) [km]` ~., data = df_train)
#SVM so numericos
modelov2 <- svm(`Range (WLTP) [km]` ~., data = df_train)
#SVM numericos + categoricos
modelov3 <- svm(`Range (WLTP) [km]` ~ ., data = df_train2, kernel = "linear", scale = TRUE)
#lasso
modelov4 <- train(`Range (WLTP) [km]` ~., data = df_train2, method = 'glmnet', tuneGrid = expand.grid(alpha = 1, lambda = 1))
#Ridge
modelov5 <- train(`Range (WLTP) [km]` ~., data = df_train2, method = 'glmnet', tuneGrid = expand.grid(alpha = 0, lambda = 1))

#Predi??o 1
summary(modelov1)
modelov1$coefficients
class(modelov1)

pred <- predict(modelov1, df_teste)

#Predi??o 2
summary(modelov2)

pred2 <- predict(modelov2, df_teste)

#Predi??o 3
summary(modelov3)
pred3 <- predict(modelov3, df_teste2)

#predi??o 4
summary(modelov4)
pred4 <- predict(modelov4, df_teste2)

#Predi??o 5
summary(modelov5)
pred5 <- predict(modelov5, df_teste2)

#M?tricas (R2, RMSE, Valores previstos)

R2models <- data.frame(
  lm = R2(pred,df_teste$`Range (WLTP) [km]` ),
  svmNum = R2(pred2,df_teste$`Range (WLTP) [km]` ),
  svmNumcat = R2(pred3,df_teste2$`Range (WLTP) [km]` ),
  Lasso = R2(pred4,df_teste2$`Range (WLTP) [km]` ),
  Ridge = R2(pred5,df_teste2$`Range (WLTP) [km]` ),
  rtree = R2(pred6,df_teste2$`Range (WLTP) [km]` )
)

RMSEmodels <- data.frame(
  lm = RMSE(pred,df_teste$`Range (WLTP) [km]` ),
  svmNum = RMSE(pred2,df_teste$`Range (WLTP) [km]` ),
  svmNumcat = RMSE(pred3,df_teste2$`Range (WLTP) [km]` ),
  Lasso = RMSE(pred4,df_teste2$`Range (WLTP) [km]` ),
  Ridge = RMSE(pred5,df_teste2$`Range (WLTP) [km]` )
)

resultmodels <- data.frame(
  testenumcat = df_teste2$`Range (WLTP) [km]`,
  lm = pred,
  lm_dif = pred - df_teste$`Range (WLTP) [km]`,
  svmNum = pred2,
  svmNum_dif = pred2 - df_teste$`Range (WLTP) [km]`,
  svmNumcat = pred3,
  svmNumcat_dif = pred3 - df_teste2$`Range (WLTP) [km]`,
  Lasso = pred4,
  Lasso_dif = pred4 - df_teste2$`Range (WLTP) [km]`,
  Ridge = pred5,
  Ridge_dif = pred5 - df_teste2$`Range (WLTP) [km]`)

R2models
RMSEmodels
summary(resultmodels)

#Agora ser? realizado o ajuste dos parametros para encontrar a melhor combina??o dos hiperparametros (Tune) para as regress?es LASSO e RIDGE
param <- c(seq(0.1, 2, by =0.1) ,  seq(2, 5, 0.5) , seq(5, 25, 1))

lasso <- train(`Range (WLTP) [km]` ~., data = df_train2, method = 'glmnet', tuneGrid = expand.grid(alpha = 1, lambda = param) , metric =  "Rsquared") 
ridge <- train(`Range (WLTP) [km]` ~., data = df_train2, method = 'glmnet', tuneGrid = expand.grid(alpha = 0, lambda = param), metric =  "Rsquared") 
linear <- train(`Range (WLTP) [km]` ~., data = df_train2, method = 'lm', metric =  "Rsquared")

print(paste0('Melhor parametro Lasso: ' , lasso$finalModel$lambdaOpt))
print(paste0('Melhor parametro Ridge: ' , ridge$finalModel$lambdaOpt))

pred_lasso <- predict(lasso, df_teste2)
pred_ridge <- predict(ridge, df_teste2)
pred_lin <- predict(linear, df_teste2)

R2_reg <- data.frame(
  Ridge_R2 = R2(pred_ridge, df_teste2$`Range (WLTP) [km]`),
  Lasso_R2 = R2(pred_lasso, df_teste2$`Range (WLTP) [km]`),
  Linear_R2 = R2(pred_lin, df_teste2$`Range (WLTP) [km]`)
)

RMSE_reg <- data.frame(
  Ridge_RMSE = RMSE(pred_ridge, df_teste2$`Range (WLTP) [km]`) , 
  Lasso_RMSE = RMSE(pred_lasso, df_teste2$`Range (WLTP) [km]`) , 
  Linear_RMSE = RMSE(pred_lin, df_teste2$`Range (WLTP) [km]`)
)


result_reg <- data.frame(
  testenumcat = df_teste2$`Range (WLTP) [km]`,
  Lasso = pred_lasso,
  Lasso_dif = pred_lasso - df_teste2$`Range (WLTP) [km]`,
  Ridge = pred_ridge,
  Ridge_dif = pred_ridge - df_teste2$`Range (WLTP) [km]`,
  lm = pred_lin,
  lm_dif = pred_lin - df_teste$`Range (WLTP) [km]`)

R2_reg
RMSE_reg
summary(result_reg)

#Conclus?o
'O modelo de regress?o LASSO e SVM foram os que demonstraram melhor acuracia dentre os modelos. Por?m como o dataset possui poucos dados,
o modelo ficou tendencioso, com risco de sofrer overfitting. Uma das propostas seria aumentar a quantidade de dados, para melhor
confiabilidade na predi??o da variavel target "RANGE (WLTP) [km]".'