#3a
CAhousing <- read.csv("/Users/baur/Downloads/CAhousing.csv")
CAhousing$AveBedrms <- CAhousing$totalBedrooms/CAhousing$households
CAhousing$AveRooms <- CAhousing$totalRooms/CAhousing$households
CAhousing$AveOccupancy <- CAhousing$population/CAhousing$households
logMedVal <- log(CAhousing$medianHouseValue)
CAhousing <- CAhousing[,-c(4,5,9)] # drop medval and room totals
CAhousing$logMedVal <- logMedVal
print(CAhousing)
#b

library(gamlr)

# Model matrix 
X <- model.matrix(logMedVal ~ . * longitude * latitude, data = CAhousing)[,-1]
y <- CAhousing$logMedVal

cvfit <- cv.gamlr(X, y)

cvfit$lambda.min
cvfit$lambda.1se

# Plot cross-validated MSE
plot(cvfit)
title("Cross-Validated MSE vs Lambda")

# Plot Lasso regularization path
plot(cvfit$gamlr)
title("Lasso Coefficient Path")

#c (i)

library(tree)

#Tree
big_tree <- tree(logMedVal ~ ., data = CAhousing,
mindev = 0,  
mincut = 10)       


plot(big_tree, type = "uniform")
text(big_tree, pretty = 0, cex = 0.6)


#c (ii) 
# Cost-complexity pruning curve
cv_tree <- cv.tree(big_tree)

plot(cv_tree$size, cv_tree$dev, type = "b",
     xlab = "Tree Size (number of terminal nodes)",
     ylab = "Deviance",
     main = "Cost-Complexity Pruning Curve")

#c (iii)
best_size <- cv_tree$size[which.min(cv_tree$dev)]
best_size   

# Prune to optimal size
pruned_tree <- prune.tree(big_tree, best = best_size)

plot(pruned_tree, type = "uniform")
text(pruned_tree, pretty = 0, cex = 0.8)

pruned_tree

#d (i)

library(gamlr)
library(tree)
library(randomForest)

MSE <- list(LASSO = NULL, CART = NULL, RF = NULL)

#ii

for (i in 1:10) {
  
  train_idx <- sample(1:nrow(CAhousing), 5000)
  
  X_train <- X_full[train_idx, ]
  y_train <- y_full[train_idx]
  
  X_test  <- X_full[-train_idx, ]
  y_test  <- y_full[-train_idx]
  
  data_train <- CAhousing[train_idx, ]
  data_test  <- CAhousing[-train_idx, ]
  
  
#Lasso
  lasso_cv  <- cv.gamlr(X_train, y_train)
  yhat_lasso <- predict(lasso_cv, X_test)
  
  # root sum of squared prediction errors
  rss_lasso <- sum((yhat_lasso - y_test)^2)
  err_lasso <- sqrt(rss_lasso)
  
  MSE$LASSO <- c(MSE$LASSO, err_lasso)
  
  
 #Cart
  big_tree <- tree(
    logMedVal ~ .,
    data = data_train,
    mindev = 0,
    mincut = 10
  )
  
  cv_t <- cv.tree(big_tree)
  best_size <- cv_t$size[which.min(cv_t$dev)]
  
  pruned_tree <- prune.tree(big_tree, best = best_size)
  yhat_cart <- predict(pruned_tree, newdata = data_test)
  
  rss_cart <- sum((yhat_cart - y_test)^2)
  err_cart <- sqrt(rss_cart)
  
  MSE$CART <- c(MSE$CART, err_cart)
  

  # Random Forest
 
  rf_fit <- randomForest(
    logMedVal ~ .,
    data = data_train,
    ntree = 500
  )
  
  yhat_rf <- predict(rf_fit, newdata = data_test)
  
  rss_rf <- sum((yhat_rf - y_test)^2)
  err_rf <- sqrt(rss_rf)
  
  MSE$RF <- c(MSE$RF, err_rf)
}

# Look at the 10 errors for each method
print(MSE)

#e

boxplot(
  as.data.frame(MSE),
  main = "Out-of-Sample RMSE Comparison Across 10 Random Splits",
  ylab = "RMSE",
  xlab = "Model",
  col = c("lightblue", "lightgreen", "lightpink")
)

