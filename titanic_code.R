library(tidyverse)
library(caret)
library(rpart)


list.files(path = "C:/Users/Ryan's Laptop/Documents/titanic")
train_data <- read_csv("C:/Users/Ryan's Laptop/Documents/titanic/train.csv")
test_data <- read_csv("C:/Users/Ryan's Laptop/Documents/titanic/test.csv")

head(train_data)
head(test_data)

## Descriptive Stats ##
table(train_data$Pclass)
table(train_data$Survived)
table(train_data$Age)
table(train_data$Embarked)
table(train_data$SibSp)
table(train_data$Parch)

table(is.na(train_data$Embarked))

table(ifelse(train_data$Fare>150,1,0))

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

# look at last names and see if families survive 


## Feature Engineering ##
train_data$Embarked[train_data$Embarked==""] = "0"
train_data$Embarked[train_data$Embarked=="S"] <- "1"
train_data$Embarked[train_data$Embarked=="Q"] <- "2"
train_data$Embarked[train_data$Embarked=="C"] <- "3"

#change to my code 
full$FamilySize <-full$SibSp + full$Parch + 1 
full$FamilySized[full$FamilySize == 1] <- 'Single' 
full$FamilySized[full$FamilySize < 5 & full$FamilySize >= 2] <- 'Small' 
full$FamilySized[full$FamilySize >= 5] <- 'Big' 
full$FamilySized=as.factor(full$FamilySized)

##Engineer features based on all the passengers with the same ticket
ticket.unique <- rep(0, nrow(full))

tickets <- unique(full$Ticket)
for (i in 1:length(tickets)) {
  current.ticket <- tickets[i]
  party.indexes <- which(full$Ticket == current.ticket)
  for (k in 1:length(party.indexes)) {
    ticket.unique[party.indexes[k]] <- length(party.indexes)
  }
}

full$ticket.unique <- ticket.unique
full$ticket.size[full$ticket.unique == 1]   <- 'Single'
full$ticket.size[full$ticket.unique < 5 & full$ticket.unique>= 2]   <- 'Small'
full$ticket.size[full$ticket.unique >= 5]   <- 'Big'

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


#Create Mother indicator
train_set <- train_set %>% mutate(Mother = ifelse(Age_new > 18 & Sex=="female" & Parch >= 1, "Mother", "Not"))
test_set <- test_set %>% mutate(Mother = ifelse(Age_new > 18 & Sex=="female" & Parch >= 1, "Mother", "Not"))
test_data <- test_data %>% mutate(Mother = ifelse(Age_new > 18 & Sex=="female" & Parch >= 1, "Mother", "Not"))

#Create Father indicator
train_set <- train_set %>% mutate(Father = ifelse(Age_new > 18 & Sex=="male" & Parch >= 1, "Father", "Not"))
test_set <- test_set %>% mutate(Father = ifelse(Age_new > 18 & Sex=="male" & Parch >= 1, "Father", "Not"))
test_data <- test_data %>% mutate(Father = ifelse(Age_new > 18 & Sex=="male" & Parch >= 1, "Father", "Not"))

#Check to see if this is any better at predicting survival?
prop.table(table(train_set$Sex, train_set$Survived), 1) #by sex if they survived 
prop.table(table(train_set$Mother, train_set$Survived), 1) #didnt help predict better 
prop.table(table(train_set$Father, train_set$Survived), 1) #didnt help predict better


#preprocess for the models you want to run after feature engineering
#it is important for models like knn
preProc <- preProcess(test_set[, c("Age_new", "SibSp", "Parch", "Fare")], method=c("center", "scale"))
test_set <- predict(preProc, test_set)
preProc2 <- preProcess(train_set[, c("Age_new", "SibSp", "Parch", "Fare")], method=c("center", "scale"))
train_set <- predict(preProc2, train_set)
preProc3 <- preProcess(test_data[, c("Age_new", "SibSp", "Parch", "Fare")], method=c("center", "scale"))
test_data <- predict(preProc3, test_data)
stepwise1 <- glm(Survived ~ factor(Pclass) + Age_new + Mother + Father + SibSp + Sex + Parch + Fare
                 + factor(Pclass)*Sex, data=train_set, family = "binomial")
library(MASS)
#vif(stepwise)
bestModel1 <- stepAIC(stepwise1, direction="both")
summary(bestModel1)
survived_hat <- predict(bestModel1, test_set, type="response")
survived_pred <- factor(ifelse(survived_hat >0.5, 1, 0))
head(survived_pred)
confusionMatrix(survived_pred, factor(test_set$Survived)) #didnt really help that much



#dont need this but just to make sense for the code i borrowed below
feauter1<-full[1:891, c("Pclass", "title","Sex","Embarked","FamilySized","ticket.size")]
response <- as.factor(train$Survived)
feauter1$Survived=as.factor(train$Survived)
###For Cross validation purpose will keep 20% of data aside from my orginal train set
##This is just to check how well my data works for unseen data
set.seed(500)
ind=createDataPartition(feauter1$Survived,times=1,p=0.8,list=FALSE)
train_val=feauter1[ind,]
test_val=feauter1[-ind,]


## Create a model##
#run a MARS model

#linear regression model
#use stepwise selection (in the kaggle slides), smaller AIC is better 
fit <- glm(Survived ~ factor(Pclass) + Sex + factor(Pclass)*Sex, data = train_set, family = "binomial")
summary(fit)

fit_2 <- glm(Survived ~  Fare + factor(Pclass) + factor(Pclass)*Fare, data = train_set, family = "binomial")
summary(fit_2)

survived_hat <- predict(fit, test_set, type="response")
survive_pred <- factor(ifelse(survived_hat > 0.5, 1, 0))
confusionMatrix(survive_pred, factor(test_set$Survived))

##decision tree
set.seed(1234)
Model_DT=rpart(Survived~.,data=train_val,method="class")
rpart.plot(Model_DT,extra =  3,fallen.leaves = T)
PRE_TDT=predict(Model_DT,data=train_val,type="class")
confusionMatrix(PRE_TDT,train_val$Survived)
#cross validation below 
PRE_VDTS=predict(Model_CDT$finalModel,newdata=test_val,type="class")
confusionMatrix(PRE_VDTS,test_val$Survived)


#Ada Boost

#random forest 
library(randomForest)

#Knn, not done 
train_knn <- train(Apps ~., method = "knn", data = train_set,
                   tuneGrid = data.frame(k=seq(1,71,2)))

ggplot(train_knn)
train_knn$bestTune #7 neighbors
y_hat <- predict(train_knn, test_set, type = "raw")

#LDA 

#FDA

#Neural Network (maybe)


## Create output file ##
survive_test <- predict(fit, test_data, type = "response") #0 to 1 prediction of surival
prediction <- factor(ifelse(survive_test > 0.5, 1, 0))
submission <- test_data %>% select(PassengerId) %>% mutate(Survived = prediction)
head(submission)
write.csv(submission, "submission.csv", row.names=FALSE)
