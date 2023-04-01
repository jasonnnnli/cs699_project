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



get_performance_measures <- function(train, test, method) {
  
  # Train the model
  model <- train(SeekCancerInfo ~ ., data = train, method = method, trControl = train_control)
  
  # Make predictions on the test set
  test_pred <- predict(model, newdata = test)
  
  # Create confusion matrix
  conf_matrix <- confusionMatrix(table(test_pred, test$SeekCancerInfo))
  
  # Calculate TP rate, FP rate, precision, recall, F-measure, ROC area, and MCC for each class
  TP_rate <- conf_matrix$byClass[,"Sensitivity"]
  FP_rate <- conf_matrix$byClass[,"Specificity"]
  precision <- conf_matrix$byClass[,"Pos Pred Value"]
  recall <- conf_matrix$byClass[,"Recall"]
  F_measure <- conf_matrix$byClass[,"F1"]
  ROC_area <- roc(test$SeekCancerInfo, as.numeric(test_pred))$auc
  MCC <- conf_matrix$overall["MCC"]
  
  # Calculate weighted averages
  weighted_TP_rate <- conf_matrix$overall["Sensitivity"]
  weighted_FP_rate <- conf_matrix$overall["Specificity"]
  weighted_precision <- conf_matrix$overall["Pos Pred Value"]
  weighted_recall <- conf_matrix$overall["Prevalence"]
  weighted_F_measure <- conf_matrix$overall["F1"]
  weighted_ROC_area <- roc(test$SeekCancerInfo, as.numeric(test_pred), levels=c(0,1))$auc
  weighted_MCC <- conf_matrix$overall["MCC"]
  
  # Return the performance measures as a list
  measures <- list(conf_matrix = conf_matrix,
                   TP_rate = TP_rate,
                   FP_rate = FP_rate,
                   precision = precision,
                   recall = recall,
                   F_measure = F_measure,
                   ROC_area = ROC_area,
                   MCC = MCC,
                   weighted_TP_rate = weighted_TP_rate,
                   weighted_FP_rate = weighted_FP_rate,
                   weighted_precision = weighted_precision,
                   weighted_recall = weighted_recall,
                   weighted_F_measure = weighted_F_measure,
                   weighted_ROC_area = weighted_ROC_area,
                   weighted_MCC = weighted_MCC)
  return(measures)
}

# Set the seed for reproducibility
set.seed(123)

# Create a list of the 5 classification algorithms to use
methods <- c("glm", "rf", "rpart", "svmRadial", "knn")

# Create a list to store the performance measures for each model
performance_measures <- list()


for (i in 1:5) {
  
  # Choose an attribution selection method and create a reduced training dataset and a reduced test dataset
  # replace this with your own code to create the reduced datasets
  reduced_train <- reduced_train_ig  
  reduced_test <- reduced_test_ig
  
  # Loop through the 5 classification algorithms and train and test the models on the reduced datasets
  for (j in 1:5) {
    
    # Get the performance measures for the model
    measures <- get_performance_measures(reduced_train, reduced_test, methods[j])
    
    # Add the performance measures to the list
    performance_measures[[paste0("Method_", methods[j], "_Reduction_", i)]] <- measures
  }
}
















