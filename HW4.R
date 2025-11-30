# HW 4
# Edison Zhong, Bia, Lara, Bauryzhan, Chloe
# Group 8

## Problem 1 

set.seed(0)
setwd("~/Downloads/Fall2025")

install.packages("pROC")
library(pROC)              

email <- read.csv("spam.csv")

logit_full <- glm(spam ~ ., data=email, family=binomial)

email$pred_prob <- predict(logit_full, type="response")

library(pROC)

roc_full <- roc(response = email$spam,
                predictor = email$pred_prob,
                plot = TRUE,
                print.auc = TRUE,
                col = "blue")

n <- nrow(email)
train_index <- sample(1:n, size = floor(0.8 * n), replace = FALSE)

train <- email[train_index, ]
test  <- email[-train_index, ]

logit_oos <- glm(spam ~ ., data=train, family=binomial)

test$pred_prob <- predict(logit_oos, newdata=test, type="response")

roc(response = test$spam,
    predictor = test$pred_prob,
    plot = TRUE,
    print.auc = TRUE)


library(tidyverse)
dw_data <- read.csv("dw_data.csv")


## Problem 2

#a) linear model, ATE 
data <- read.csv("/Users/larali/ECON460/data files/dw_data.csv")

Y <- data$re78
D <- data$treated

model <-glm(Y~D+age+education+black+hispanic+married+re74+re75+ue74+ue75, data=data)
summary(model)
ate <- coef(summary(model))["DTRUE","Estimate"]
se <- coef(summary(model))["DTRUE","Std. Error"]
ate
se

# b) propensity scores, ATE 
# binary logit 
ps_model <- glm(treated ~ age + education + black + hispanic + married +
                  re74 + re75 + ue74 + ue75,
                data = data,
                family = binomial())
# keep 
ps <- predict(ps_model, type = "response")
keep <- ps >= 0.05 & ps <= 0.95
data_trim <- data[keep, ]
ps_trim   <- ps[keep]
Y_trim    <- data_trim$re78
D_trim    <- data_trim$treated
# calculate Ytilde 
w <- ifelse(D_trim == 1, 1 / ps_trim, 1 / (1 - ps_trim))
Y_tilde <- w * Y_trim
# OLS
model_ipw <- lm(Y_tilde ~ D_trim)
summary(model_ipw)
# ATE
ATE_ipw <- mean(Y_tilde)
ATE_ipw


# c) 
#regression tree
library(tree)
data$D_fac <- factor(D)
ps_tree <- tree(D_fac ~ age + education + black + hispanic + married +
                  re74 + re75 + ue74 + ue75,
                data = data)
print(ps_tree)
plot(ps_tree)
text(ps_tree, pretty = 0)

pred_tree <- predict(ps_tree, type = "vector")
ps_tree_hat <- pred_tree[, "TRUE"]   # P(D=TRUE | X)

keep_tree <- ps_tree_hat >= 0.05 & ps_tree_hat <= 0.95
Y_tree   <- Y[keep_tree]
D_tree   <- D[keep_tree]
ps_treet <- ps_tree_hat[keep_tree]

w_tree <- ifelse(D_tree == TRUE, 1/ps_treet, 1/(1 - ps_treet))
Y_tilde_tree <- w_tree * Y_tree

model_ipw_tree <- lm(Y_tilde_tree ~ D_tree)
summary(model_ipw_tree)
ATE_tree <- mean(Y_tilde_tree)
ATE_tree

#random forest
library(ranger)

data$D_fac <- factor(D, levels = c(FALSE, TRUE))  

ps_ranger <- ranger(
  D_fac ~ age + education + black + hispanic + married +
    re74 + re75 + ue74 + ue75,
  data        = data,
  probability = TRUE,
  num.trees   = 500,
  mtry        = 3,
  min.node.size = 5
)

pred_ranger <- predict(ps_ranger, data = data)
ps_ranger_hat <- pred_ranger$predictions[, "TRUE"]

keep_rf <- ps_ranger_hat >= 0.05 & ps_ranger_hat <= 0.95
Y_rf   <- Y[keep_rf]
D_rf   <- D[keep_rf]
ps_rft <- ps_ranger_hat[keep_rf]

w_rf <- ifelse(D_rf == TRUE, 1/ps_rft, 1/(1 - ps_rft))

Y_tilde_rf <- w_rf * Y_rf

model_ipw_rf <- lm(Y_tilde_rf ~ D_rf)
summary(model_ipw_rf)

ATE_rf <- mean(Y_tilde_rf)
ATE_rf

# e)
install.packages("MatchIt")
library(MatchIt)
library(dplyr)

m.out <- matchit(
  treated ~ age + education + black + hispanic + married +
    re74 + re75 + ue74 + ue75,
  data = data,
  method = "nearest"
)
matched_data <- match.data(m.out)
model_match <- lm(
  re78 ~ treated + factor(subclass),
  data = matched_data,
  weights = weights
)

summary(model_match)
ATE_match <- coef(summary(model_match))["treatedTRUE", "Estimate"]
SE_match  <- coef(summary(model_match))["treatedTRUE", "Std. Error"]

ATE_match
SE_match

## Problem 3 
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
best_size    # <-- Report this number in your write-up

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


## Problem 4 
#q 4
# Load required libraries
library(dplyr)
library(ggplot2)

# 4(a) Read the data files
members <- read.csv("C:\\Users\\rawre\\Downloads\\rollcall-members.csv")
votes <- read.csv("C:\\Users\\rawre\\Downloads\\rollcall-votes.csv")

# take look at data
head(votes)
head(members)
head(votes[, 1:10])
# -1 means 'nay', +1 means 'yea', 0 means 'abstained'

# 4(b) PCA with K = 10 and analyze results

# prepare voting data for PCA (remove any non-vote columns if present)
# assumes the first column might be identifiers
vote_matrix <- as.matrix(votes[, -1]) # remove first column if it's identifiers

# Run PCA with 10 components
pca_result <- prcomp(vote_matrix, center = TRUE, scale. = TRUE, rank. = 10)

# scree plot
plot(pca_result, main = "Scree Plot for Congressional Votes PCA")

# proportion of variance explained
summary_pca <- summary(pca_result)
print(summary_pca)

# extract variance explained by first two components
var_explained <- summary_pca$importance[2, ] # Proportion of Variance row
cat("Variance explained by PC1:", round(var_explained[1]*100, 2), "%\n")
cat("Variance explained by PC2:", round(var_explained[2]*100, 2), "%\n")
cat("Cumulative variance explained by PC1 and PC2:", 
    round(sum(var_explained[1:2])*100, 2), "%\n")

# 4(c) plot first 2  principal components colored by party

# get estimated latent factors
factors <- predict(pca_result)

# convert to data frame for plotting
plot_data <- data.frame(
  PC1 = factors[, 1],
  PC2 = factors[, 2],
  party = members$party,
  member = members$member
)

# plotting
ggplot(plot_data, aes(x = PC1, y = PC2, color = party)) +
  geom_point(alpha = 0.7, size = 2) +
  scale_color_manual(values = c("D" = "blue", "R" = "red", "DR" = "purple")) +
  labs(title = "First Two Principal Components of Congressional Votes",
       x = paste0("PC1 (", round(var_explained[1]*100, 1), "% variance)"),
       y = paste0("PC2 (", round(var_explained[2]*100, 1), "% variance)"),
       color = "Party") +
  theme_minimal() +
  theme(legend.position = "bottom")

# interpretate first principal component
# Examine the loadings to understand PC1
pc1_loadings <- pca_result$rotation[, 1]
top_loadings <- head(sort(abs(pc1_loadings), decreasing = TRUE), 10)
cat("Top 10 loadings for PC1:\n")
print(top_loadings)

# 4(d) examine extreme values on PC1 to support interpretation

# add PC1 scores to member data for analysis
members_with_pc <- members %>%
  mutate(PC1 = factors[, 1])

# t5 largest PC1 values (most extreme in one direction)
cat("5 members with largest PC1 values:\n")
top_pc1 <- members_with_pc %>%
  arrange(desc(PC1)) %>%
  head(5)
print(top_pc1[, c("member", "state", "party", "PC1")])

# t5 smallest PC1 values (most extreme in opposite direction)
cat("\n5 members with smallest PC1 values:\n")
bottom_pc1 <- members_with_pc %>%
  arrange(PC1) %>%
  head(5)
print(bottom_pc1[, c("member", "state", "party", "PC1")])

# party distribution among extremes
cat("\nParty distribution for top 5 PC1:\n")
table(top_pc1$party)

cat("\nParty distribution for bottom 5 PC1:\n")
table(bottom_pc1$party)

# additional analysis of overall party means on PC1
cat("\nMean PC1 by party:\n")
members_with_pc %>%
  group_by(party) %>%
  summarize(mean_PC1 = mean(PC1, na.rm = TRUE),
            n_members = n())
