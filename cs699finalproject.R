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
library(tidyr)
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
df <- subset(df, select = -c(Treatment_H5C4,Pandemic, Language_Flag,DRA,Stratum))
df
# Split the dataset into training and test sets

set.seed(31)
split <- initial_split(df, prop = 0.66, strata = SeekCancerInfo)
train <- training(split)
test <- testing(split)

set.seed(31)
# repeat 10-fold cross-validation 
train_control <- trainControl(method = "cv", number = 10, 
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






compute_performance_measures <- function(true_labels, predictions) {
  
  confusion <- confusionMatrix(true_labels, predictions)
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



rpart_tuneGrid <- expand.grid(cp = seq(0.01, 0.5, length.out = 10))
svm_tuneGrid <- expand.grid(sigma = 0.1, C = 1)

models <- list(
  J48 = list(method = "J48"),
  rpart = list(method = "rpart", tuneGrid = rpart_tuneGrid),
  nnet = list(method = "nnet"),
  rf = list(method = "rf"),
  svmRadial = list(method = "svmRadial", tuneGrid = svm_tuneGrid)
)

reduced_datasets <- list(
  IG = list(train = reduced_train_ig, test = reduced_test_ig),
  GR = list(train = reduced_train_gr, test = reduced_test_gr),
  Gini = list(train = reduced_train_gini, test = reduced_test_gini),
  ReliefK = list(train = reduced_train_Reliefk, test = reduced_test_Reliefk),
  Euclid = list(train = reduced_train_Euclid, test = reduced_test_Euclid)
)

combined_performance_measures <- data.frame()

# Iterate over the feature selection methods
for (reduced_data_name in names(reduced_datasets)) {
  reduced_train <- reduced_datasets[[reduced_data_name]]$train
  reduced_test <- reduced_datasets[[reduced_data_name]]$test
  
  # Iterate over the models
  for (model_name in names(models)) {
    model_params <- models[[model_name]]
    
    model <- train(SeekCancerInfo ~ ., data = reduced_train, method = model_params$method, trControl = train_control, tuneGrid = model_params$tuneGrid)
    test_pred <- predict(model, newdata = reduced_test)
    
    true_labels <- reduced_test$SeekCancerInfo
    predictions <- test_pred
    performance_measures <- compute_performance_measures(true_labels, predictions)
    
    # Add the model name and feature selection method as new columns to the performance_measures dataframe
    performance_measures$Model <- model_name
    performance_measures$FeatureSelection <- reduced_data_name
    
    # Combine the performance measures of each model into a single dataframe
    combined_performance_measures <- rbind(combined_performance_measures, performance_measures)
  }
}

# Spread the combined_performance_measures data frame to have separate columns for each model
spread_performance_measures <- combined_performance_measures %>%
  unite("Model_FeatureSelection", Model, FeatureSelection) %>%
  spread(Model_FeatureSelection, Value)

# Print the spread performance measures
cat("\n\nSpread performance measures:\n")
print(spread_performance_measures)

###best model  J48_Gini
model <- train(SeekCancerInfo ~ ., data = reduced_train_gini, method = "J48", trControl = train_control)
model
test_pred <- predict(model, newdata = reduced_test_gini)
confusionMatrix(test_pred, reduced_test_gini$SeekCancerInfo)

#####
model <- train(SeekCancerInfo ~ ., data = train, method = "J48", trControl = train_control)
model
test_pred <- predict(model, newdata = test)
confusionMatrix(test_pred, test$SeekCancerInfo)



#####dataset 
write.csv(df, "preprocessed_dataset.csv", row.names = FALSE)
write.csv(train, "initial_train_dataset.csv", row.names = FALSE)
write.csv(test, "initial_test_dataset.csv", row.names = FALSE)
write.csv(reduced_train_gini, "best_model_train_dataset.csv", row.names = FALSE)
write.csv(reduced_test_gini, "best_model_test_dataset.csv", row.names = FALSE)


