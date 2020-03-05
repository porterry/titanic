library(tidyverse)
library(caret)


list.files(path = "C:/Users/Ryan's Laptop/Documents/titanic")
train_data <- read_csv("C:/Users/Ryan's Laptop/Documents/titanic/train.csv")
test_data <- read_csv("C:/Users/Ryan's Laptop/Documents/titanic/test.csv")

head(train_data)
head(test_data)

#Create test & train datasets 
test_index <- createDataPartition(train_data$Survived, times=1,p=0.5, list=FALSE)
test_set <- train_data[test_index,]
train_set <- train_data[-test_index,]

#Descriptive Stats
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

#feature Engineering 
train_data$Embarked[train_data$Embarked==""] = "0"
train_data$Embarked[train_data$Embarked=="S"] <- "1"
train_data$Embarked[train_data$Embarked=="Q"] <- "2"
train_data$Embarked[train_data$Embarked=="C"] <- "3"


#Create a model
fit <- glm(Survived ~ factor(Pclass) + Sex + factor(Pclass)*Sex, data = train_set, family = "binomial")
summary(fit)

fit_2 <- glm(Survived ~  Fare + factor(Pclass) + factor(Pclass)*Fare, data = train_set, family = "binomial")
summary(fit_2)

survived_hat <- predict(fit, test_set, type="response")
survive_pred <- factor(ifelse(survived_hat > 0.5, 1, 0))
confusionMatrix(survive_pred, factor(test_set$Survived))

#Create output file
survive_test <- predict(fit, test_data, type = "response") #0 to 1 prediction of surival
prediction <- factor(ifelse(survive_test > 0.5, 1, 0))
submission <- test_data %>% select(PassengerId) %>% mutate(Survived = prediction)
head(submission)
write.csv(submission, "submission.csv", row.names=FALSE)
