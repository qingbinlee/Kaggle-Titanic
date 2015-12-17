## Libraries
library(rpart)
library(randomForest)
library(class)
library(e1071)
library(xgboost)
library(gbm)

## Load Data
setwd("~/Dropbox/Kaggle/Titanic- Machine Learning from Disaster")
train <- read.csv("Data/train.csv")
test <- read.csv("Data/test.csv")
test$Survived <- 0                    # Even column size
combine <- rbind(train, test)
set.seed(110)

#### HANDLE MISSING DATA
combine$Embarked[c(62,830)] = "S"                       # Interpolate NA Embarked as mode ("S")
combine$Fare[1044] <- median(combine$Fare, na.rm = T)   # Interpolate NA Fare as median

# Interpolate NA age via tree 
Agefit <- rpart(Age ~ Pclass + Sex + Fare + Embarked + SibSp + Parch, data = combine[!is.na(combine$Age),], method = "anova")
combine$Age[is.na(combine$Age)] <- predict(Agefit, combine[is.na(combine$Age),])


#### VARIABLES

## Make child variable
combine$Child <- 0
combine$Child[combine$Age < 18] <- 1

## Name information
combine$Name = as.character(combine$Name)                  # turn back to character

combine$Title <- sapply(combine$Name, FUN = function(x) {strsplit(x, split='[,.]')[[1]][2]})
combine$Title <- sub(' ', '', combine$Title)
combine$Title[combine$Title %in% c('Mme', 'Mlle')] <- 'Mlle'
combine$Title[combine$Title %in% c('Capt', 'Don', 'Major', 'Sir')] <- 'Sir'
combine$Title[combine$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] <- 'Lady'
combine$Title <- as.factor(combine$Title)


## Family Size
combine$FamilySize <- combine$SibSp + combine$Parch + 1

## Parentless Child
combine$Parentless <- 0
combine$Parentless[combine$Child == 1 & combine$Parch == 0] <- 1

## Cabin information
combine$Cabin = as.character(combine$Cabin)                                        # turn back to character
combine$CabinL = sapply(combine$Cabin, FUN = function(x) { substring(x, 1, 1) })   # get first letter of cabin
combine$CabinL <- factor(combine$CabinL)                                           # turn back into factor

# Family ID for large families 
combine$Surname <- sapply(combine$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][1]})
combine$FamilyID <- paste(as.character(combine$FamilySize), combine$Surname, sep="")
combine$FamilyID[combine$FamilySize <= 2] <- 'Small'
famIDs <- data.frame(table(combine$FamilyID))
famIDs <- famIDs[famIDs$Freq <= 2,]
combine$FamilyID[combine$FamilyID %in% famIDs$Var1] <- 'Small'
combine$FamilyID <- factor(combine$FamilyID)

combine$FamilyID2 <- combine$FamilyID
combine$FamilyID2 <- as.character(combine$FamilyID2)
combine$FamilyID2[combine$FamilySize <= 3] <- 'Small'
combine$FamilyID2 <- factor(combine$FamilyID2)


#### MODELS

## Resplit Train and Test
train <- combine[1:891,]
test <- combine[892:1309,]

## "Woman and children first" Model
fit <- lm(Survived ~ Sex + Child, data = train)
summary(fit)

## "W&CF + Class" Model
fit2 <- lm(Survived ~ Sex + Child + Pclass, data = train)
summary(fit2)

## Decision Tree Model
fit3 <- rpart(Survived ~ Pclass + Sex + Age + FamilySize + Fare + Embarked + Title + FamilyID, data=train, method="class")
Pred3 <- predict(fit3, test, type = "class")

# Random Forest Model
fit4 <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + Fare + Embarked + Title + FamilyID2 + FamilySize + SibSp + Parch, data=train, importance=T, ntree=2000)
Pred4 <- predict(fit4, test, type = "class")

# SVM Model
fit5 <- svm(as.factor(Survived) ~ Pclass + Sex + Age + Child + Fare + Embarked + Title + FamilySize + FamilyID + SibSp + Parch, data = train)
Pred5 <- predict(fit5, test, type = "class")
Pred5b <- predict(fit5, train, type = "class")
table(train$Survived, Pred5b)

# Gradient Boosting model
fit.gbm <- gbm(Survived ~ Pclass + Sex + Age + Child + Fare + Embarked + Title + FamilySize + FamilyID + SibSp + Parch,
             data=train,
             distribution="gaussian",
             n.trees=1000,
             shrinkage=0.005,
             bag.fraction=0.7,
             interaction.depth=3)
preds<-predict(fit.gbm, test, OOB=TRUE, type='response', n.trees=1000)
normPreds<-(preds-min(preds))/(max(preds)-min(preds))
Pred7 <- as.numeric(normPreds > 0.5)

# Ensemble Model (RPart, Random Forest, SVM -- majority vote)
Pred6 <- Pred5
Pred6[Pred5 != Pred4 & Pred5 != Pred3 & Pred6 == 0] <- 1
Pred6[Pred5 != Pred4 & Pred5 != Pred3 & Pred6 == 1] <- 0
# Pred6[Pred5 != Pred4 & Pred5 != Pred7 & Pred6 == 0] <- 1
# Pred6[Pred5 != Pred4 & Pred5 != Pred7 & Pred6 == 1] <- 0


#### SUBMIT
submit <- data.frame(PassengerId = test$PassengerId, Survived = Pred6)
write.csv(submit, file = "submission.csv", row.names = FALSE)
