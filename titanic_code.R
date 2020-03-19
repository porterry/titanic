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


## linear regression model
#use stepwise selection (in the kaggle slides), smaller AIC is better 
fit <- glm(Survived ~ factor(Pclass) + Sex + factor(Pclass)*Sex, data = train_set, family = "binomial")
summary(fit)

fit_2 <- glm(Survived ~  Fare + factor(Pclass) + factor(Pclass)*Fare, data = train_set, family = "binomial")
summary(fit_2)
survived_hat <- predict(fit, test_set, type="response")
survive_pred <- factor(ifelse(survived_hat > 0.5, 1, 0))
confusionMatrix(survive_pred, factor(test_set$Survived))


stepwise1 <- glm(Survived ~ factor(Pclass) + Age_new + Sex + FamilySized, data=train_set, family = "binomial")
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


#Ada Boost

#random forest 
library(randomForest)
fit <- randomForest(margin ~ .,data=polls_2008)
plot(fit)

polls_2008 %>% mutate(y_hat = predict(fit, polls_2008)) %>%
  ggplot() + 
  geom_point(aes(day, margin)) + 
  geom_line(aes(day, y_hat),col="red")

#node is the minium value to split
train_rf <- randomForest(y ~ ., data=mnist_27$train)
confusionMatrix(predict(train_rf, mnist_27$test), mnist_27$test$y) #0.785

install.packages("Rborist")
train_rf2 <- train(y ~., method="Rborist", 
                   tuneGrid=data.frame(predFixed=2, minNode=c(3,50)),
                   data=mnist_27$train)

confusionMatrix(predict(train_rf_2, mnist_27$test), mnist_27$test$y)

#
rf <- randomForest(extractFeatures(train), as.factor(train$Survived), ntree=100, importance=TRUE)

submission <- data.frame(PassengerId = test$PassengerId)
submission$Survived <- predict(rf, extractFeatures(test))
write.csv(submission, file = "1_random_forest_r_submission.csv", row.names=FALSE)


# Knn
train_knn <- train(Survived ~ Age_new + Sex + factor(Pclass), method = "knn", data = train_set,
                   tuneGrid = data.frame(k=seq(1,71,2)))

plot(train_knn)
train_knn$bestTune #13 neighbors
y_hat <- predict(train_knn, test_set, type = "raw")
survived_pred <- factor(ifelse(y_hat >0.5, 1, 0))
head(survived_pred)
confusionMatrix(survived_pred, factor(test_set$Survived)) 

#FDA
#run a MARS model

#Neural Network (maybe)


## Create output file ##
survive_test <- predict(fit, test_data, type = "response") #0 to 1 prediction of surival
prediction <- factor(ifelse(survive_test > 0.5, 1, 0))
submission <- test_data %>% select(PassengerId) %>% mutate(Survived = prediction)
head(submission)
write.csv(submission, "submission.csv", row.names=FALSE)
