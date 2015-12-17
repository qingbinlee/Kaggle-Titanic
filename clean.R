# Libraries
library(rpart)


# Load Data
setwd("~/Dropbox/Kaggle/Titanic- Machine Learning from Disaster")
train <- read.csv("Data/train.csv", stringsAsFactors=FALSE)
test <- read.csv("Data/test.csv", stringsAsFactors=FALSE)
test$Survived <- 0
combine <- rbind(train, test)
set.seed(110)


# Handle missing data
combine$Fare[is.na(combine$Fare)] <- median(combine$Fare, na.rm = T) # Replace NA Fare with median
combine$Embarked[combine$Embarked == ""] = "S"                       # Raplace NA Embarked with mode ("S")
# Use tree to replace NA age  
Agefit <- rpart(Age ~ Pclass + Sex + Fare + Embarked + SibSp + Parch, data = combine[!is.na(combine$Age),], method = "anova")
combine$Age[is.na(combine$Age)] <- predict(Agefit, combine[is.na(combine$Age),])


# change data type
combine$Survived = as.factor(combine$Survived)
combine$Pclass = as.factor(combine$Pclass)
combine$Sex = as.factor(combine$Sex)
combine$Embarked = as.factor(combine$Embarked)


# Create New Variables

# Whether the passage is a child
combine$Child <- 0
combine$Child[combine$Age < 18] <- 1
combine$Child = as.factor(combine$Child)

# Name information
#combine$Name = as.character(combine$Name)
combine$Title <- sapply(combine$Name, FUN = function(x) {strsplit(x, split='[,.]')[[1]][2]})
combine$Title <- sub(' ', '', combine$Title)
combine$Title[combine$Title %in% c('Mme', 'Mlle')] <- 'Mlle'
combine$Title[combine$Title %in% c('Capt', 'Don', 'Major', 'Sir')] <- 'Sir'
combine$Title[combine$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] <- 'Lady'
combine$Title <- as.factor(combine$Title)

# Fare level
combine$FareLevel <- '30+'
combine$FareLevel[combine$Fare < 30 & combine$Fare >= 20] <- '20-30'
combine$FareLevel[combine$Fare < 20 & combine$Fare >= 10] <- '10-20'
combine$FareLevel[combine$Fare < 10] <- '<10'

# Family size
combine$FamilySize <- combine$SibSp + combine$Parch + 1
combine$Surname <- sapply(combine$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][1]})
combine$FamilyID <- paste(as.character(combine$FamilySize), combine$Surname, sep="")
combine$FamilyID[combine$FamilySize <= 2] <- 'Small'

famIDs <- data.frame(table(combine$FamilyID))
famIDs <- famIDs[famIDs$Freq <= 2,]
combine$FamilyID[combine$FamilyID %in% famIDs$Var1] <- 'Small'
combine$FamilyID <- factor(combine$FamilyID)

# Number of people in family
combine$Family <- combine$SibSp + combine$Parch + 1

# # Parentless Child
# combine$Parentless <- 0
# combine$Parentless[combine$Child == 1 & combine$Parch == 0] <- 1

## Cabin information
combine$Cabin = as.character(combine$Cabin)             
combine$CabinLevel = sapply(combine$Cabin, FUN = function(x) { substring(x, 1, 1) })
combine$CabinLevel <- factor(combine$CabinL)


#### MODELS

## Resplit Train and Test
train <- combine[1:891,]
test <- combine[892:1309,]
write.csv(train, file = "train.csv", row.names = FALSE)
write.csv(test, file = "test.csv", row.names = FALSE)
# 
# ## "Woman and children first" Model
# fit <- lm(Survived ~ Sex + Child, data = train)
# summary(fit)
# 
# ## "W&CF + Class" Model
# fit2 <- lm(Survived ~ Sex + Child + Pclass, data = train)
# summary(fit2)
# 
# ## Decision Tree Model
# fit3 <- rpart(Survived ~ Pclass + Sex + Age + Fare + Embarked + Title, data=train, method="class")
# Pred3 <- predict(fit3, test, type = "class")
# 
# # Random Forest Model
# fit4 <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + Fare + Embarked + Title + SibSp + Parch, data=train, importance=T, ntree=2000)
# Pred4 <- predict(fit4, test, type = "class")
# 
# # SVM Model
# fit5 <- svm(as.factor(Survived) ~ Pclass + Sex + Age + Child + Fare + Embarked + Title + SibSp + Parch, data = train)
# Pred5 <- predict(fit5, test, type = "class")
# Pred5b <- predict(fit5, train, type = "class")
# table(train$Survived, Pred5b)
# 
# # Ensemble Model (RPart, Random Forest, SVM -- majority vote)
# Pred6 <- Pred5
# Pred6[Pred5 != Pred4 & Pred5 != Pred3 & Pred6 == 0] <- 1
# Pred6[Pred5 != Pred4 & Pred5 != Pred3 & Pred6 == 1] <- 0
# 
# 
# #### SUBMIT
# submit <- data.frame(PassengerId = test$PassengerId, Survived = Pred6)
# write.csv(submit, file = "submission.csv", row.names = FALSE)
