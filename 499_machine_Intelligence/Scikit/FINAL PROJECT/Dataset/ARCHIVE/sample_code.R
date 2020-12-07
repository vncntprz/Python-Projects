test = read.csv("Psychopath_Testset_v1.csv")

set.seed(48484)
submission1 = data.frame(myID = test$myID, psychopathy = sample(1:nrow(test)))
write.csv(submission1, file = "./upload/random1.csv", row.names = FALSE) 



train = read.csv("Psychopath_Trainingset_v1.csv")

#function for adding NAs indicators to dataframe and replacing NA's with a value---"cols" is vector of columns to operate on
#   (necessary for randomForest package)
appendNAs <- function(dataset, cols) {
  append_these = data.frame( is.na(dataset[, cols] ))
  names(append_these) = paste(names(append_these), "NA", sep = "_")
  dataset = cbind(dataset, append_these)
  dataset[is.na(dataset)] = -1
  return(dataset)
}

#replacements:
train <- appendNAs(train,3:ncol(train))
test <- appendNAs(test,3:ncol(test))

library("randomForest")
rf = randomForest(train[,3:ncol(train)],train$psychopathy, do.trace=TRUE,importance=TRUE, sampsize = nrow(train)*.7, ntree = 300)
predictions = predict(rf, test[,3:ncol(test)])
submission3 = data.frame(myID = test$myID, psychopathy = predictions)
write.csv(submission3, file = "./upload/random_forest.csv", row.names = FALSE) 





