library(caret)
library(rsample)  
#SeekCancerInfo  AgeGrpA  EducA  BMI  HHInc  smokeStat PHQ4 AvgDrinksPerWeek
data <-read.csv('C:/Users/Jason/Downloads/hints5_cycle4_public (2).csv')

get_mode <- function(x) {
  unique_x <- unique(x)
  unique_x[which.max(tabulate(match(x, unique_x)))]
}

preProcess <- c("SeekCancerInfo", "AgeGrpA","EducA","HHInc", "smokeStat","PHQ4", "AvgDrinksPerWeek")

# Replace negative values with the mode in specified columns
for (column in preProcess) {
  non_negative_data <- data[data[[column]] >= 0, column]
  mode_value <- get_mode(non_negative_data)
  data[data[[column]] < 0, column] <- mode_value
}

data[data[["BMI"]] < 0, "BMI"] <- mean( data[["BMI"]])
print(data$BMI)
data$SeekCancerInfo
data$SeekCancerInfo <- factor(data$SeekCancerInfo)


set.seed(31)
split <- initial_split(data, prop = 0.66, strata = SeekCancerInfo)
train <- training(split)
test <- testing(split)
modelLookup("J48")

set.seed(31)
# repeat 10-fold cross-validation 5 times
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, 
                              summaryFunction = defaultSummary)

## use tuneGrid

J48Grid <-  expand.grid(C = c(0.01, 0.25, 0.5), M = (1:4))
model <- train(SeekCancerInfo ~ ., data = train, method = "J48", trControl = train_control,
               tuneGrid = J48Grid
)
model
test_pred <- predict(model, newdata = test)
confusionMatrix(test_pred, test$class)