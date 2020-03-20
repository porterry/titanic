library(tidyverse)
library(caret)
library(rpart)
library(VIM)

list.files(path = "C:/Users/Ryan/Documents/titanic")
train_data <- read_csv("C:/Users/Ryan/Documents/titanic/train.csv")
test_data <- read_csv("C:/Users/Ryan/Documents/titanic/test.csv")

head(train_data)
head(test_data)

## Descriptive Stats ##
table(train_data$Pclass)
table(train_data$Survived)
table(train_data$Age)
table(train_data$Embarked)
table(train_data$SibSp)
table(train_data$Parch)

aggr(train_data, prop = FALSE, combined = TRUE, numbers = TRUE, sortVars = TRUE, sortCombs = TRUE)

train_data %>%
  ggplot(aes(Age)) +
  geom_bar(stat = "count")

train_data %>%
  ggplot(aes(Pclass, Fare)) +
  geom_boxplot(aes(group = Pclass))

train_data %>%
  ggplot(aes(Survived, Age)) +
  geom_boxplot(aes(group = Survived))

train_data %>%
  ggplot(aes(Parch, Age)) +
  geom_boxplot(aes(group = Parch))

## Feature Engineering ##
table(train_data$Embarked)
table(is.na(train_data$Embarked))
table(is.na(test_data$Embarked))
train_data <- train_data %>% mutate(Embarked = ifelse(is.na(Embarked), "S", Embarked))

#family size
train_data$FamilySize <-train_data$SibSp + train_data$Parch + 1 
train_data$FamilySized[train_data$FamilySize == 1] <- 'Single' 
train_data$FamilySized[train_data$FamilySize < 5 & train_data$FamilySize >= 2] <- 'Small' 
train_data$FamilySized[train_data$FamilySize >= 5] <- 'Big' 
train_data$FamilySized=as.factor(train_data$FamilySized)

table(train_data$FamilySized)

test_data$FamilySize <-test_data$SibSp + test_data$Parch + 1 
test_data$FamilySized[test_data$FamilySize == 1] <- 'Single' 
test_data$FamilySized[test_data$FamilySize < 5 & test_data$FamilySize >= 2] <- 'Small' 
test_data$FamilySized[test_data$FamilySize >= 5] <- 'Big' 
test_data$FamilySized=as.factor(test_data$FamilySized)

table(test_data$FamilySized)

##Engineer features based on title
train_data <- mutate(train_data, title_orig = factor(str_extract(Name, "[A-Z][a-z]*\\.")))
test_data <- mutate(test_data, title_orig = factor(str_extract(Name, "[A-Z][a-z]*\\.")))

table(train_data$title_orig)
table(test_data$title_orig)

## Create test & train datasets ##
test_index <- createDataPartition(train_data$Survived, times=1,p=0.5, list=FALSE)
test_set <- train_data[test_index,]
train_set <- train_data[-test_index,]

#predict ages into misses values
#log sibsip + 1, because it has a big negative sibsip right scewed 
library(naniar)
miss_var_summary(train_set)

linearMod <- lm(Age ~ Pclass + Sex + log(SibSp+1) + Parch + Fare, data=train_set)
#look at the model
summary(linearMod)
#does it meet the assumptions of a linear model? How do the residuals look
plot(linearMod)
#check other assumptions of linear model
library(car)
dwt(linearMod) #check for independent errors - want values close to 2.
vif(linearMod) #check the variance inflation factor - values greater than 10 are problematic
#predict the age based on the model
Age_pre_train <- round(predict(linearMod, train_set))
Age_pre_test_set <- round(predict(linearMod, test_set))
Age_pre_test_data <- round(predict(linearMod, test_data))
#set our ages for all three sets
train_set <- train_set %>% mutate(Age_new = ifelse(is.na(Age), Age_pre_train, Age))
test_set <- test_set %>% mutate(Age_new = ifelse(is.na(Age), Age_pre_test_set, Age))
test_data <- test_data %>% mutate(Age_new = ifelse(is.na(Age), Age_pre_test_data, Age))


#preprocess for the models you want to run after feature engineering
#it is important for models like knn
preProc <- preProcess(test_set[, c("Age_new", "SibSp", "Parch", "Fare")], method=c("center", "scale"))
test_set <- predict(preProc, test_set)
preProc2 <- preProcess(train_set[, c("Age_new", "SibSp", "Parch", "Fare")], method=c("center", "scale"))
train_set <- predict(preProc2, train_set)
preProc3 <- preProcess(test_data[, c("Age_new", "SibSp", "Parch", "Fare")], method=c("center", "scale"))
test_data <- predict(preProc3, test_data)


## Create a model##


## linear regression model, done
#use stepwise selection (in the kaggle slides), smaller AIC is better 
fit <- glm(Survived ~ factor(Pclass) + Sex + factor(Pclass)*Sex, data = train_set, family = "binomial")
summary(fit)

fit_2 <- glm(Survived ~  Fare + factor(Pclass) + factor(Pclass)*Fare, data = train_set, family = "binomial")
summary(fit_2)
survived_hat <- predict(fit, test_set, type="response")
survive_pred <- factor(ifelse(survived_hat > 0.5, 1, 0))
confusionMatrix(survive_pred, factor(test_set$Survived))


stepwise1 <- glm(Survived ~ factor(Pclass) + Age_new + Sex + FamilySized + Embarked, 
                 data=train_set, 
                 family = "binomial")
library(MASS)
#vif(stepwise)
bestModel1 <- stepAIC(stepwise1, direction="both")
summary(bestModel1)
survived_hat <- predict(bestModel1, test_set, type="response")
survived_pred <- factor(ifelse(survived_hat >0.5, 1, 0))
confusionMatrix(survived_pred, factor(test_set$Survived)) #82.74%


## decision tree
set.seed(1234)
Model_DT=rpart(Survived~.,data=train_set,method="class")
rpart.plot(Model_DT,extra =  3,fallen.leaves = T)
PRE_TDT=predict(Model_DT,data=train_set,type="class")
confusionMatrix(PRE_TDT,train_set$Survived)
#cross validation below 
PRE_VDTS=predict(Model_CDT$finalModel,newdata=test_val,type="class")
confusionMatrix(PRE_VDTS,test_val$Survived)


#Ada Boost, done
library(fastAdaboost)

train_ada <- train(factor(Survived) ~ factor(Pclass) + Age_new + Sex + FamilySized + Embarked,
                   data = train_set,
                   method = "adaboost",
                   tuneGrid = data.frame(nIter = seq(1,101,5), method = "adaboost"))

ggplot(train_ada, highlight = TRUE)
y_hat_ada_prob <- predict(train_ada, test_set, type = "prob")
y_hat_ada <- predict(train_ada, test_set, type = "raw")
confusionMatrix(y_hat_ada, factor(test_set$Survived)) #82.51

#random forest, done 
library(randomForest)

train_rf <- train(factor(Survived) ~  factor(Pclass) + Age_new + Sex + FamilySized + Embarked,
                  method = "rf",
                  data=train_set,
                  tuneGrid = data.frame(mtry=seq(1,14,2)))

y_hat_rf_prob <-predict(train_rf, test_set, type="prob")
y_hat_rf <- predict(train_rf, test_set)
confusionMatrix(y_hat_rf, factor(test_set$Survived)) #81.61


# Knn, done
train_knn <- train(factor(Survived) ~ Age_new + Sex + factor(Pclass) + FamilySized, 
                   method = "knn", 
                   data = train_set,
                   tuneGrid = data.frame(k=seq(1,71,2)))

plot(train_knn)
train_knn$bestTune #13 neighbors
y_hat_knn_prob <- predict(train_knn, test_set, type = "prob")
y_hat_knn <- predict(train_knn, test_set)
confusionMatrix(y_hat_knn, factor(test_set$Survived)) #80.72%

#FDA, done 
#run a MARS model
library(mda)
train_fda <- train(factor(Survived) ~ Age_new + Sex + factor(Pclass) + FamilySized, 
                   method = "fda",
                   data = train_set)

ggplot(train_fda, highlight = TRUE)
y_hat_fda <- predict(train_fda, test_set, type = "raw")
confusionMatrix(y_hat_fda, factor(test_set$Survived)) #79.15

#Neural Network (maybe)
library(neuralnet)
f <- as.formula('Survived ~ Sex + Age_new + FamilySized + Pcalss')
set.seed(8)
fit <- neuralnet(f,train_set,hidden=5,linear.output=F)
plot(fit)
p <- compute(fit,testData); p <- ifelse(p[[2]]>0.5,1,0)

submit <- data.frame(PassengerId=892:1309,Survived=p)
write.csv(submit,'TitanicDeepNet.csv',row.names=F)
x = which( (test$Sex=='male' & p==1) | (test$Sex=='female' & p==0) )
row.names(test) <- 892:1309; test[x,c('Name','Sex','Age','SibSp','Parch','FareAdj','OneSurvived','AllDied')]


nn <- neuralnet(train_set$Survived ~ factor(Pclass) + Age_new + Sex + FamilySized + Embarked, 
                data = train_set[,-1], hidden = 5, linear.output = F)

# plot network
plot(nn, rep="best")

library(caret)
predict <- compute(nn, xtrain.df[,-c(1,9,10)])
predicted.class=apply(predict$net.result,1,which.max)-1
confusionMatrix( as.factor(predicted.class), as.factor(xtrain$Survived))


#Support Vector Machine, done
caret_svm <- train(Survived ~ factor(Pclass) + Age_new + Sex + FamilySized + Embarked, 
                   data=train_set, method='svmRadial',  
                   trControl=trainControl(method="cv", number=5))
caret_svm
solution_svm <- predict(caret_svm, test_set)
survived_svm <- factor(ifelse(solution_svm >0.5, 1, 0))
confusionMatrix(survived_svm, factor(test_set$Survived)) #81.61%

# Averaging Predictors
ens_avg_prob <- (y_hat_ada_prob[,2] + y_hat_knn_prob[,2] + y_hat_rf_prob[,2]) / 3
ens_avg <- as.factor(ifelse(ens_avg_prob > 0.5, 1, 0))
head(ens_avg)
confusionMatrix(ens_avg, factor(test_set$Survived)) #.82


## Create output file ##
survive_test <- predict(bestModel1, test_data, type = "response") #0 to 1 prediction of surival
prediction <- factor(ifelse(survive_test > 0.5, 1, 0))
submission <- test_data %>% dplyr::select(PassengerId) %>% mutate(Survived = prediction)
head(submission)
write.csv(submission, "submission.csv", row.names=FALSE)

survive_test <- predict(train_ada, test_data, type = "prob") #0 to 1 prediction of surival
prediction <- factor(ifelse(survive_test > 0.5, 1, 0))
submission_ada <- test_data %>% dplyr::select(PassengerId) %>% mutate(Survived = prediction)
head(submission_ada)
write.csv(submission_ada, "submission_ada.csv", row.names=FALSE)


y_hat_ada_prob <- predict(train_ada, test_set, type = "prob")
y_hat_ada <- predict(train_ada, test_set, type = "raw")
confusionMatrix(y_hat_ada, factor(test_set$Survived)) #82.51