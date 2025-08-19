heart_disease <- read.csv("C:/Users/ASUS/Downloadsheart_disease33.txt", sep="")
library(summarytools)
library(dplyr)
library(tidyverse)
library(alr4)
library(class)
library(caret)
library(tree)
library(MASS)
library(randomForest)
library(ROCR)
library(gbm)
library(nnet)
library(pROC)

#Check for Missing data
str(heart_disease)
summary(heart_disease)
freq(heart_disease)
freq(heart_disease$BMI)

n=nrow(heart_disease)
##Drop NAs for all categorical variables, no NAs for continuous variable
heart_disease1 <- heart_disease %>% 
  drop_na(HeartDiseaseorAttack, Stroke, Diabetes) #create a dataset without NAs without change variable type
heart_disease2 <- heart_disease %>% 
  drop_na(HeartDiseaseorAttack, Stroke, Diabetes) #create a dataset without NAs without change variable type
#Change categorical variables from integer to factor
heart_disease1$HeartDiseaseorAttack <- as.factor(heart_disease1$HeartDiseaseorAttack)
heart_disease1$Stroke <- as.factor(heart_disease1$Stroke)
heart_disease1$Diabetes <- as.factor(heart_disease1$Diabetes)
summary(heart_disease1)

##Iterative Imputation for missing values
heart_disease3 <- heart_disease
heart_disease3$HeartDiseaseorAttack <- as.factor(heart_disease3$HeartDiseaseorAttack)
heart_disease3$Stroke <- as.factor(heart_disease3$Stroke)
heart_disease3$Diabetes <- as.character(heart_disease3$Diabetes)
summary(heart_disease3)


heart_disease4 = heart_disease3
##Impute the missings in HeartDiseaseorAttack, Stroke, Diabetes using most frequent category
heart_disease3$HeartDiseaseorAttack[is.na(heart_disease3$HeartDiseaseorAttack)]="0"
heart_disease3$Stroke[is.na(heart_disease3$Stroke)]="0"
heart_disease3$Diabetes[is.na(heart_disease3$Diabetes)]="0"
summary(heart_disease3)

n_iter=10 #Set number of iterations
for(i in 1:n_iter)
{
  #impute Stroke given rest
  m_Stroke <- glm(Stroke~., data=heart_disease3, subset=!is.na(heart_disease$Stroke), family=binomial)
  pred_Stroke <- predict(m_Stroke, heart_disease3[is.na(heart_disease$Stroke),], type="response")
  heart_disease3$Stroke[is.na(heart_disease$Stroke)] <-  ifelse(pred_Stroke>0.5,1,0)
  #impute HeartDiseaseorAttack given rest
  m_HDA <- glm(HeartDiseaseorAttack~.,heart_disease3,subset=!is.na(heart_disease$HeartDiseaseorAttack), family=binomial)
  pred_HDA <- predict(m_HDA, heart_disease3[is.na(heart_disease$HeartDiseaseorAttack),], type="response")
  heart_disease3$HeartDiseaseorAttack[is.na(heart_disease$HeartDiseaseorAttack)] <- ifelse(pred_HDA>0.5,1,0)
  #impute Diabetes given rest
  m_Diabetes <- multinom(Diabetes~., heart_disease3, subset=!is.na(heart_disease$Diabetes), trace=FALSE)
  pred_Diabetes <- predict(m_Diabetes, heart_disease3[is.na(heart_disease$Diabetes),], type="class")
  heart_disease3$Diabetes[is.na(heart_disease$Diabetes)] = pred_Diabetes
}
summary(heart_disease3)

#Logistic Regression model for HeartDiseaseorAttack 
logm1 <- glm(HeartDiseaseorAttack~., data=heart_disease1, family="binomial")
summary(logm1)

pchisq(logm1$deviance, 5691, lower.tail=FALSE) #deviance test for good fit

Pearson = sum(residuals(logm1, type="pearson")^2) #chi-square test for good fit
Pearson

pchisq(Pearson, 5691, lower.tail=FALSE)

plot(logm1, which=5)#Diagnosis Plot

pi_logit_d1 <- predict(logm1, type="response")
y_logit_d1 <- ifelse(pi_logit_d1 >0.5, 1, 0)
ER_logit_d1 <- mean(y_logit_d1!=heart_disease1$HeartDiseaseorAttack)#Error Rate
##ROC curve and AUC
pred_logit_d1 <- prediction(pi_logit_d1, heart_disease1$HeartDiseaseorAttack)
perf_logit_d1 <- performance(pred_logit_d1, "tpr","fpr")
plot(perf_logit_d1, colorize=TRUE, main="Missing Data impute by dropping NAs")

AUC_logit_d1 <- performance(pred_logit_d1,"auc")@y.values[[1]]


##logistic regression model on iterative regression impute
logm2 <- glm(HeartDiseaseorAttack~., data=heart_disease3, family="binomial")
summary(logm2)

pchisq(logm2$deviance, 5691, lower.tail=FALSE) #deviance test for good fit
Pearson = sum(residuals(logm2, type="pearson")^2)#chi-square test for good fit
Pearson

pchisq(Pearson, 5691, lower.tail=FALSE)
plot(logm2, which=5) #Diagnosis Plot

pi_logit_d2 <- predict(logm2, type="response")
y_logit_d2 <- ifelse(pi_logit_d2 >0.5, 1, 0)
ER_logit_d2 <- mean(y_logit_d2!=heart_disease3$HeartDiseaseorAttack) #Error Rate
##ROC Curve
pred_logit_d2 <- prediction(pi_logit_d2, heart_disease3$HeartDiseaseorAttack)
perf_logit_d2 <- performance(pred_logit_d2, "tpr","fpr")
plot(perf_logit_d2, colorize=TRUE, main="Missing Data impute by Iterative Regression")

AUC_logit_d2 <- performance(pred_logit_d2,"auc")@y.values[[1]]
##Comparsion between two datasets
data.frame(model=c("Dropping NAs RF","Iterative Regression RF"),ER=c(ER_logit_d1,ER_logit_d2))
par(mfrow=c(1,2))
plot(perf_logit_d1, colorize=TRUE, main="Dropping NAs logit")
plot(perf_logit_d2, colorize=TRUE, main="Iterative Regression logit")

data.frame(model=c("Dropping NAs RF","Iterative Regression RF"),AUC=c(AUC_logit_d1,AUC_logit_d2))

##Split the dataset into 50% training, and 50% validation
set.seed(4052)
train_ind <- sample(1:nrow(heart_disease1), round(0.5*nrow(heart_disease1)))
train_ind <- sample(1:nrow(heart_disease1), round(0.5*nrow(heart_disease1)))
train <- heart_disease1[train_ind,]
valid <- heart_disease1[-train_ind,] 

#Use KNN to predict HeartDiseaseorAttack
train2 <- heart_disease2[train_ind,] 
valid2 <- heart_disease2[-train_ind,] 
train.knn <- train2[,-1]
val.knn <- valid2[,-1]
y_train <- as.numeric(train2[,1])
y_test <- as.numeric(valid2[,1])

#Use KNN to predict HeartDiseaseorAttack on iterative regression impute
train_ind2 <- sample(1:nrow(heart_disease3), round(0.5*nrow(heart_disease3)))
train3 <- heart_disease3[train_ind2,]
valid3 <- heart_disease3[-train_ind2,]
train.knn2 <- train3[,-1]
val.knn2 <- valid3[,-1]
y_train2 <- as.numeric(train3[,1])-1
y_test2 <- as.numeric(valid3[,1])-1

##Use 10-folds Cross-validation to choose K, with a list of k=1,10,70,100
klist <- c(1,10,70,100)
nfolds <- 10
fold=createFolds(1:nrow(train2),k=nfolds,list=FALSE)
kCV_err <- rep(0,4)
for(i in 1:nfolds){
  pre1.CV <- knn(train.knn[fold!=i,], val.knn[fold!=i,], cl=train2$HeartDiseaseorAttack[fold!=i],k=1)
  pre10.CV <- knn(train.knn[fold!=i,], val.knn[fold!=i,], cl=train2$HeartDiseaseorAttack[fold!=i],k=10)
  pre70.CV <- knn(train.knn[fold!=i,], val.knn[fold!=i,], cl=train2$HeartDiseaseorAttack[fold!=i],k=70)
  pre100.CV <- knn(train.knn[fold!=i,], val.knn[fold!=i,], cl=train2$HeartDiseaseorAttack[fold!=i],k=100)
  kCV_err[1]=kCV_err[1]+mean(pre1.CV!=valid$HeartDiseaseorAttack[fold==i])/nfolds
  kCV_err[2]=kCV_err[2]+mean(pre10.CV!=valid$HeartDiseaseorAttack[fold==i])/nfolds
  kCV_err[3]=kCV_err[3]+mean(pre70.CV!=valid$HeartDiseaseorAttack[fold==i])/nfolds
  kCV_err[4]=kCV_err[4]+mean(pre100.CV!=valid$HeartDiseaseorAttack[fold==i])/nfolds
}

data.frame(klist, kCV_err) #Use k=70 since greater k does not lead to change in ER

k1 <- knn(train.knn, val.knn, cl=train2[,1], k=70, prob=TRUE)
table(valid$HeartDiseaseorAttack,k1)

val_err <- mean(valid$HeartDiseaseorAttack!=k1)
val_err

##ROC Curves and Error Rate, and AUC
winning_pi_knn_d1 <- as.numeric(attr(k1, "prob"))
winning_class <- k1
pi_knn_d1 <- ifelse(winning_class==1, winning_pi_knn_d1, 1-winning_pi_knn_d1)
y_knn_d1 <- ifelse(pi_knn_d1 >0.5, 1, 0)
ER_knn_d1 <- mean((y_test-y_knn_d1)^2) # Error Rate

pred_knn_d1 <- prediction(pi_knn_d1, y_test)
perf_knn_d1 <- performance(pred_knn_d1, "tpr","fpr")
par(mfrow=c(1,1))
plot(perf_knn_d1, colorize=TRUE, main="Dropping NAs KNN") #ROC Curve d1


AUC_knn_d1 <- performance(pred_knn_d1,"auc")@y.values[[1]]
##Use 10-folds Cross-validation to choose y_train ##Use 10-folds Cross-validation to choose K, with a list of k=1, 10, 70, 100 on iterative model
klist2 <- c(1,10,70,100)
nfolds <- 10
fold2=createFolds(1:nrow(train2),k=nfolds,list=FALSE)
kCV_err2 <- rep(0,4)
for(i in 1:nfolds){
  pre12.CV <- knn(train.knn2[fold2!=i,], val.knn2[fold2!=i,], cl=train3$HeartDiseaseorAttack[fold2!=i],k=1)
  pre102.CV <- knn(train.knn2[fold2!=i,], val.knn2[fold2!=i,], cl=train3$HeartDiseaseorAttack[fold2!=i],k=10)
  pre702.CV <- knn(train.knn2[fold2!=i,], val.knn2[fold2!=i,], cl=train3$HeartDiseaseorAttack[fold2!=i],k=70)
  pre1002.CV <- knn(train.knn2[fold2!=i,], val.knn2[fold2!=i,], cl=train3$HeartDiseaseorAttack[fold2!=i],k=100)
  kCV_err2[1]=kCV_err2[1]+mean(pre12.CV!=valid3$HeartDiseaseorAttack[fold2==i])/nfolds
  kCV_err2[2]=kCV_err2[2]+mean(pre102.CV!=valid3$HeartDiseaseorAttack[fold2==i])/nfolds
  kCV_err2[3]=kCV_err2[3]+mean(pre702.CV!=valid3$HeartDiseaseorAttack[fold2==i])/nfolds
  kCV_err2[4]=kCV_err2[4]+mean(pre1002.CV!=valid3$HeartDiseaseorAttack[fold2==i])/nfolds
}

data.frame(klist2, kCV_err2) #Use k=70 since greater k does not lead to change in ER

k2 <- knn(train.knn2, val.knn2, cl=train3[,1], k=70, prob=TRUE)
table(valid3$HeartDiseaseorAttack,k2)

val_err2 <- mean(valid3$HeartDiseaseorAttack!=k2)
val_err2

##ROC Curves and Error Rate, and AUC
winning_pi_knn_d2 <- as.numeric(attr(k2, "prob"))
winning_class2 <- k2
pi_knn_d2 <- ifelse(winning_class2==1, winning_pi_knn_d2, 1-winning_pi_knn_d2)
y_knn_d2 <- ifelse(pi_knn_d2 >0.5, 1, 0)
ER_knn_d2 <- mean((y_test2-y_knn_d2)^2) # Error Rate

pred_knn_d2 <- prediction(pi_knn_d2, y_test2)
perf_knn_d2 <- performance(pred_knn_d2, "tpr","fpr")
par(mfrow=c(1,1))
plot(perf_knn_d2, colorize=TRUE, main="Iterative Regression KNN")#ROC Curve d2


AUC_knn_d2 <- performance(pred_knn_d2,"auc")@y.values[[1]]
##Compare between two datasets
data.frame(model=c("Dropping NAs RF","Iterative Regression RF"),ER=c(ER_knn_d1,ER_knn_d2))

par(mfrow=c(1,2))
plot(perf_knn_d1, colorize=TRUE, main="Dropping NAs KNN") #ROC Curve d1
plot(perf_knn_d2, colorize=TRUE, main="Iterative Regression KNN") #ROC Curve d2

data.frame(model=c("Dropping NAs RF","Iterative Regression RF"),AUC=c(AUC_knn_d1,AUC_knn_d2))

#Random Forest
##Use 10-fold Cross-validation to choose mtry value with m=1,2,3,4,5,6,7
nfolds <- 10
fold3=createFolds(1:nrow(train), k=nfolds, list=FALSE)
mlist <- c(1,2,3,4,5,6,7)
ertemp <- rep(NA, 10)
erlist <- rep(NA, 7)

for(i in 1:length(mlist)){
  mvalue <- mlist[i]
  for(j in 1:nfolds){
    rf_temp <- randomForest(HeartDiseaseorAttack~., data=train[fold3==j,], mtry=mvalue, ntree=2000, importance=TRUE)
    ertemp[j] <- mean(rf_temp$err.rate[,1])
  }
  erlist[i] = mean(ertemp)
}
data.frame(mlist,erlist)#m=1 is selected

## Fit random Forest model with mtry=1
rf <- randomForest(HeartDiseaseorAttack~., data=heart_disease1, mtry=1, ntree=2000, importance=TRUE)
rf

varImpPlot(rf)
##Error Rate and ROC Curve, and AUC
pred_rf_d1 <- predict(rf, heart_disease1, type="response")
ER_rf_d1 <- mean(heart_disease1$HeartDiseaseorAttack!=pred_rf_d1)
pi_rf_d1 <- rf$votes[,2]
pred2_rf_d1 <- prediction(pi_rf_d1, heart_disease1$HeartDiseaseorAttack)
perf_rf_d1 <- performance(pred2_rf_d1, "tpr","fpr")
par(mfrow=c(1,1))
plot(perf_rf_d1, colorize=TRUE, main="Dropping NAs RF") # ROC Curve for d1


AUC_rf_d1 <- performance(pred2_rf_d1,"auc")@y.values[[1]]

##Use 10-fold Cross-validation to choose mtry value with m=1,2,3,4,5,6,7 on iterative regression impute
nfolds <- 10
fold4=createFolds(1:nrow(train3), k=nfolds, list=FALSE)
mlist2 <- c(1,2,3,4,5,6,7)
ertemp2 <- rep(NA, 10)
erlist2 <- rep(NA, 7)
for(i in 1:length(mlist2)){
  mvalue2 <- mlist2[i]
  for(j in 1:nfolds){
    rf_temp2 <- randomForest(HeartDiseaseorAttack~., data=train3[fold4==j,], mtry=mvalue2, ntree=2000, importance=TRUE)
    ertemp2[j] <- mean(rf_temp2$err.rate[,1])
  }
  erlist2[i] = mean(ertemp2)
}
data.frame(mlist2,erlist2)#m=1 is selected

## Fit random Forest model with mtry=1
rf2 <- randomForest(HeartDiseaseorAttack~., data=heart_disease3, mtry=1, ntree=2000, importance=TRUE)
rf2
varImpPlot(rf2)

##Error Rate and ROC Curve, and AUC
pred_rf_d2 <- predict(rf2, heart_disease3, type="response")
ER_rf_d2 <- mean(heart_disease3$HeartDiseaseorAttack!=pred_rf_d2)
pi_rf_d2 <- rf2$votes[,2]
pred2_rf_d2 <- prediction(pi_rf_d2, heart_disease3$HeartDiseaseorAttack)
perf_rf_d2 <- performance(pred2_rf_d2, "tpr","fpr")
plot(perf_rf_d2, colorize=TRUE, main="Iterative Regression RF")# ROC Curve for d2


AUC_rf_d2 <- performance(pred2_rf_d2,"auc")@y.values[[1]]
##Compare two datasets
data.frame(model=c("Dropping NAs RF","Iterative Regression RF"),ER=c(ER_rf_d1,ER_rf_d2))
par(mfrow=c(1,2))
plot(perf_rf_d1, colorize=TRUE, main="Dropping NAs RF") # ROC Curve for d1
plot(perf_rf_d2, colorize=TRUE, main="Iterative Regression RF") # ROC Curve for d2
data.frame(model=c("Dropping NAs RF","Iterative Regression RF"),AUC=c(AUC_rf_d1,AUC_rf_d2))
