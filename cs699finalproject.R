library(caret)
library(rsample)  
library(dplyr)
library(pROC)
library(caret)
library(rsample)
library(RWeka)
library(rpart)
library(MASS)
library(RWeka)
library(OneR)
library(FSelector)
library(corpcor)

library(pROC)
#SeekCancerInfo  AgeGrpA  EducA  BMI  HHInc  smokeStat PHQ4 AvgDrinksPerWeek
df <-read.csv('C:/Users/Jason/Downloads/hints5_cycle4_public (2).csv')



get_mode <- function(x) {
  unique_x <- unique(x)
  unique_x[which.max(tabulate(match(x, unique_x)))]
}


# Replace negative values with the mode in specified columns
for (column in names(df)) {
  non_negative_data <- df[df[[column]] >= 0, column]
  mode_value <- get_mode(non_negative_data)
  df[df[[column]] < 0, column] <- mode_value
}
df$AgeGrpA
df$SeekCancerInfo <- factor(df$SeekCancerInfo)
df[] <- lapply(df, function(x) {
  if (is.character(x)) {
    x <- as.factor(x)
  }
  return(x)
})
df <- subset(df, select = -c(Treatment_H5C4,Pandemic))
df
# Split the dataset into training and test sets

set.seed(31)
split <- initial_split(df, prop = 0.66, strata = SeekCancerInfo)
train <- training(split)
test <- testing(split)

set.seed(31)
# repeat 10-fold cross-validation 
train_control <- trainControl(method = "repeatedcv", number = 10, 
                              summaryFunction = defaultSummary)



######attribute selection methods and reducedata set

library(CORElearn)


# 1. Information Gain
top_attributes_ig <- attrEval(SeekCancerInfo ~ ., train, estimator = "InfGain")
top_attributes_ig <- names(top_attributes_ig)[order(top_attributes_ig, decreasing = TRUE)][1:5]

reduced_train_ig <- train[, c(top_attributes_ig, "SeekCancerInfo")]
reduced_test_ig <- test[, c(top_attributes_ig, "SeekCancerInfo")]

# 2. Gain Ratio
top_attributes_gr <- attrEval(SeekCancerInfo ~ ., train, estimator = "GainRatio")
top_attributes_gr <- names(top_attributes_gr)[order(top_attributes_gr, decreasing = TRUE)][1:5]

reduced_train_gr <- train[, c(top_attributes_gr, "SeekCancerInfo")]
reduced_test_gr <- test[, c(top_attributes_gr, "SeekCancerInfo")]

# 3. Gini Index
top_attributes_gini <-  attrEval(SeekCancerInfo ~ ., train, estimator = "Gini")
top_attributes_gini <- names(top_attributes_gini)[order(top_attributes_gini, decreasing = TRUE)][1:5]
reduced_train_gini <- train[, c(top_attributes_gini, "SeekCancerInfo")]
reduced_test_gini <- test[, c(top_attributes_gini, "SeekCancerInfo")]


####4 ReliefFequalK

top_attributes_Reliefk <-  attrEval(SeekCancerInfo ~ ., train, estimator = "ReliefFequalK")
top_attributes_Reliefk<- names(top_attributes_Reliefk)[order(top_attributes_Reliefk, decreasing = TRUE)][1:5]
reduced_train_Reliefk <- train[, c(top_attributes_Reliefk, "SeekCancerInfo")]
reduced_test_Reliefk <- test[, c(top_attributes_Reliefk, "SeekCancerInfo")]

###5 ImpurityEuclid



top_attributes_Euclid <-  attrEval(SeekCancerInfo ~ ., train, estimator = "ImpurityEuclid")
top_attributes_Euclid<- names(top_attributes_Euclid)[order(top_attributes_Euclid, decreasing = TRUE)][1:5]
reduced_train_Euclid <- train[, c(top_attributes_Euclid, "SeekCancerInfo")]
reduced_test_Euclid <- test[, c(top_attributes_Euclid, "SeekCancerInfo")]



 
##################Information Gain  reduced_train_ig  reduced_test_ig
####################J48

modelLookup("J48")

## use tuneGrid

J48Grid <-  expand.grid(C = c(0.01, 0.25, 0.5), M = (1:4))
model <- train(SeekCancerInfo ~ ., data = train, method = "J48", trControl = train_control,
               tuneGrid = J48Grid
)
model
test_pred <- predict(model, newdata = test)





compute_performance_measures <- function(true_labels, predictions) {
  confusion <- confusionMatrix(true_labels, predictions)
  print(confusion)
  TP_rate <- confusion$byClass['Sensitivity']
  FP_rate <- confusion$byClass['Specificity']
  precision <- confusion$byClass['Pos Pred Value']
  recall <- confusion$byClass['Sensitivity']
  F_measure <- confusion$byClass['F1']
  MCC <- confusion$byClass['MCC']
  
  roc_obj <- roc(true_labels, as.numeric(predictions), levels = rev(levels(true_labels)), direction = "<")
  roc_area <- auc(roc_obj)
  
  weights <- table(true_labels) / length(true_labels)
  weighted_avg <- function(x) sum(x * weights)
  
  weighted_TP_rate <- weighted_avg(TP_rate)
  weighted_FP_rate <- weighted_avg(FP_rate)
  weighted_precision <- weighted_avg(precision)
  weighted_recall <- weighted_avg(recall)
  weighted_F_measure <- weighted_avg(F_measure)
  weighted_MCC <- weighted_avg(MCC)
  
  performance_measures <- data.frame(
    Measure = c("TP rate", "FP rate", "Precision", "Recall", "F-measure", "ROC area", "MCC",
                "Weighted TP rate", "Weighted FP rate", "Weighted Precision", "Weighted Recall",
                "Weighted F-measure", "Weighted MCC"),
    Value = c(TP_rate, FP_rate, precision, recall, F_measure, roc_area, MCC,
              weighted_TP_rate, weighted_FP_rate, weighted_precision, weighted_recall,
              weighted_F_measure, weighted_MCC)
  )
  
  
  return(performance_measures)
}




true_labels <- test$SeekCancerInfo
predictions <- test_pred
performance_measures <- compute_performance_measures(true_labels, predictions)

print(performance_measures)


#################Decision Tree



modelLookup("rpart")



## use tuneLength
model <- train(SeekCancerInfo ~ ., data = train, method = "rpart", trControl = train_control,
               tuneLength = 10)
model
test_pred <- predict(model, newdata = test)
confusionMatrix(test_pred, test$SeekCancerInfo)


####################  knn



modelLookup("knn")



knnModel <- train(SeekCancerInfo ~., data = train, method = "knn",
                  trControl=train_control,
                  preProcess = c("center", "scale"),
                  tuneLength = 100)

knnModel

test_pred <- predict(knnModel, newdata = test)
confusionMatrix(test_pred, test$SeekCancerInfo)
















##############Random Forest



library(randomForest)
library(caret)

# Define the tuning grid
grid <- expand.grid(mtry = seq(1, ncol(train) - 1, by = 2))

rf_fit <- train(SeekCancerInfo ~ ., data = train, method = "rf", trControl = train_control, tuneGrid = grid)

rf_fit$bestTune
# Output: mtry
#        9

plot(rf_fit)

# Test the model on the test dataset and generate the confusion matrix
predictions <- predict(rf_fit, newdata = test)
confusionMatrix(predictions, test$SeekCancerInfo)








############Support Vector Machine


library(e1071)
library(caret)

# Define the tuning grid
grid <- expand.grid(sigma = seq(0.1, 1, length.out = 5), C = c(0.1, 1, 10, 100))

# Perform 10-fold cross-validation using the trainControl, tuneGrid, and train functions
svm_fit <- train(SeekCancerInfo ~ ., data = train, method = "svmRadial", trControl = train_control, tuneGrid = grid)

# Check the best values of sigma and C
svm_fit$bestTune

plot(svm_fit)

# Test the model on the test dataset and generate the confusion matrix
predictions <- predict(svm_fit, newdata = test)
confusionMatrix(predictions, test$SeekCancerInfo)   
