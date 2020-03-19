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