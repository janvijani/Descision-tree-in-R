install.packages("partykit")
install.packages("rpart")
install.packages("rpart.plot")
install.packages("grid")
install.packages("libcoin")
install.packages("mvtnorm")

library(grid)
library(libcoin)
library(mvtnorm)
library(partykit)

library(rpart)
library(rpart.plot)

titanic_data$PassengerId<-NULL
titanic_data$Name<-NULL
titanic_data$Ticket<-NULL
titanic_data$Cabin<-NULL

#here, we want to convert multiple columns into factor format. So we use the following code: 

vect1<-c("Survived","Pclass","SibSp","Parch")
titanic_data[,vect1]<-lapply(titanic_data[,vect1],as.factor)
summary(titanic_data)

#impute age and embarked using their central tendencies 
titanic_data$Age[is.na(titanic_data$Age)]<-median(titanic_data$Age,na.rm = T)
titanic_data$Embarked[is.na(titanic_data$Embarked)]<-"S"
colSums(is.na(titanic_data))

titanic_tree<-ctree(Survived~., data=titanic_data)
plot(titanic_tree,type = "simple")

pred_survived<-predict(titanic_tree, titanic_data)
head(pred_survived)

table(actual=titanic_data$Survived, predicted=pred_survived)

acc<-(492+246)/nrow(titanic_data);acc #0.82
recall_1<-246/(246+96);recall_1 #0.71
recall_0<-492/(492+57); recall_0 #0.89


#Ctree: classification tree
#dependent var: categorical 
#independent VAR : cat/continuous ; regression tree

#rpart: classification as well as regression tree 

library(rpart)
library(rpart.plot)

tree_rpart<-rpart(Survived~., data= titanic_data)
rpart.plot(tree_rpart)
#the 0.38 means proportion of 1 
#hence 0.62 shows proportion of 2 
#% shows how much % of data stays in the node 

pred_survived_rpart<-predict(tree_rpart, titanic_data, type = "class")
head(pred_survived_rpart)
table(actual=titanic_data$Survived, predicted= pred_survived_rpart)

acc<-(498+244)/nrow(titanic_data);acc #0.83
recall_1<-244/(244+98);recall_1 #0.71
recall_0<-498/(498+51); recall_0 #0.90

#regression descision tree
#here, our dependent variable is continuous
#ind var is continuous or categorical
#insurance dataset where dep var=charges

#here, we use RMSE 

#implementing LM on this data:
summary(insurance)
insurance$children<-as.factor(insurance$children)
boxplot(insurance$charges)
summary(insurance$charges)
IQR(insurance$charges) # 11899.63
UW<-16640+1.5*11899.63; UW #34489.44

#hence, here we change values > upper whisker by value of upper whisker 
insurance$charges[insurance$charges>UW]<-UW
boxplot(insurance$charges)
summary(insurance$charges)

#now we create a null model and a full model 
null_insurance<-lm(charges~1, data=insurance)
full_insurance<-lm(charges~., data = insurance)

step(full_insurance,direction = "backward",
     scope = list(lower=null_insurance, upper=full_insurance))

insurance_lm<-glm(formula = charges ~ age + bmi + children + smoker + region, 
                  data = insurance)

pred_charges_lm<-predict(insurance_lm, insurance)
library(caret)
library(lattice)
library(ggplot2)

RMSE(pred_charges_lm, insurance$charges) #5086.87

#using descision tree
insurance_DT<-rpart(charges~., data=insurance)
rpart.plot(insurance_DT)
mean(insurance$charges)

pred_charges_dt<-predict(insurance_DT, insurance)
RMSE(pred_charges_dt, insurance$charges) #] 4491.466
#hence since RMSE (DT)< RMSE (LM) we can say that we get better predicted values using LM than DT for this example 

head(pred_charges_dt,20)
head(pred_charges_lm,20)

#for LM, if dep var is continuous in nature, the out put is continuous in nature 
#for DT, even if our dep var is continuous in nature, the out put is discrete in nature 
#hence, the next step for this is Random Forest 

A <- matrix(c(1,2,3,4),nrow = 2, ncol = 2)
# matrix A
A

X <- matrix(c(5,6), nrow = 2, ncol = 1)
B <- cbind(A,X)
#matrix B
B

Y <- matrix(c(7,8,9), nrow = 1, ncol = 3)
C <-rbind(B,Y)
#matrix C
C

