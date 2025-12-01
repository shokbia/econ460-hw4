# HW 4
# Edison Zhong, Bia, Lara, Bauryzhan, Chloe
# Group 8

################################################################################
### PROBLEM 1
################################################################################

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

auc_in_sample <- roc_full$auc

roc_oos <- roc(response = test$spam,
               predictor = test$pred_prob)

auc_out_sample <- roc_oos$auc

auc_in_sample
auc_out_sample


library(tidyverse)
dw_data <- read.csv("dw_data.csv")


################################################################################
### PROBLEM 2
################################################################################

library(tidyverse)
library(tree)      
library(ranger) 
library(MatchIt)

#Read the data
dw_data <- read.csv("DSCI/Fall2025/ECON460/data/dw_data.csv")

# Convert boolean/logical variables to numeric
dw_data$treated_int <- as.numeric(dw_data$treated)
dw_data$black_int <- as.numeric(dw_data$black)
dw_data$hispanic_int <- as.numeric(dw_data$hispanic)
dw_data$married_int <- as.numeric(dw_data$married)
dw_data$ue74_int <- as.numeric(dw_data$ue74)
dw_data$ue75_int <- as.numeric(dw_data$ue75)

################################################################################
#(a) Linear model with constant treatment effects
################################################################################

model <- lm(re78 ~ treated_int + age + education + black_int + hispanic_int + 
              married_int + re74 + re75 + ue74_int + ue75_int, 
            data = dw_data)

# Display full regression results
summary(model)

# Extract ATE (coefficient on treated_int)
coef_summary <- summary(model)$coefficients
ate_estimate <- coef_summary["treated_int", "Estimate"]
ate_se <- coef_summary["treated_int", "Std. Error"]
coef_summary
ate_estimate 
ate_se 

################################################################################
#(b) Propensity score weighting method with binary logit
################################################################################

# STEP 1: Estimate Propensity Scores using Binary Logit
propensity_model <- glm(treated_int ~ age + education + black_int + hispanic_int + 
                          married_int + re74 + re75 + ue74_int + ue75_int,
                        data = dw_data,
                        family = binomial(link = "logit"))

# Calculate predicted propensity scores
dw_data$propensity_score <- predict(propensity_model, type = "response")
# [Removed: propensity score summaries by group]

# STEP 2: Trim Sample - Keep observations with propensity scores in [0.05, 0.95]
dw_data_trimmed <- dw_data[dw_data$propensity_score >= 0.05 & 
                             dw_data$propensity_score <= 0.95, ]

# STEP 3: Calculate Propensity Score Weighted Outcome (Y_tilde)
dw_data_trimmed$weight <- ifelse(dw_data_trimmed$treated_int == 1,
                                 1 / dw_data_trimmed$propensity_score,
                                 1 / (1 - dw_data_trimmed$propensity_score))
dw_data_trimmed$y_tilde <- dw_data_trimmed$weight * dw_data_trimmed$re78

# STEP 4: Calculate ATE (Average of Y_tilde)
ate_ps <- mean(dw_data_trimmed$y_tilde)
ate_ps

# STEP 5: OLS Regression with Weighted Outcome
weighted_model <- lm(y_tilde ~ treated_int + age + education + black_int + 
                       hispanic_int + married_int + re74 + re75 + 
                       ue74_int + ue75_int,
                     data = dw_data_trimmed)

print(summary(weighted_model))

################################################################################
#(c) Propensity score weighting with (i) regression tree, (ii) random forests
################################################################################

# PART (i): REGRESSION TREE
tree_model <- tree(treated_int ~ age + education + black_int + hispanic_int + 
                     married_int + re74 + re75 + ue74_int + ue75_int,
                   data = dw_data)

# Get propensity scores from tree
dw_data$p_tree <- predict(tree_model, type = "vector")

# Trim to [0.05, 0.95]
dw_tree_trimmed <- dw_data[dw_data$p_tree >= 0.05 & dw_data$p_tree <= 0.95, ]

# Calculate weights and Y_tilde
dw_tree_trimmed$weight <- ifelse(dw_tree_trimmed$treated_int == 1,
                                 1 / dw_tree_trimmed$p_tree,
                                 1 / (1 - dw_tree_trimmed$p_tree))
dw_tree_trimmed$y_tilde <- dw_tree_trimmed$weight * dw_tree_trimmed$re78

# OLS with Y_tilde
tree_weighted_model <- lm(y_tilde ~ treated_int + age + education + black_int + 
                            hispanic_int + married_int + re74 + re75 + 
                            ue74_int + ue75_int,
                          data = dw_tree_trimmed)

print(summary(tree_weighted_model))
ate_tree <- mean(dw_tree_trimmed$y_tilde)
ate_tree

# PART (ii): RANDOM FORESTS
dw_data$treated_factor <- as.factor(dw_data$treated_int)

set.seed(123)
rf_model <- ranger(treated_factor ~ age + education + black_int + 
                           hispanic_int + married_int + re74 + re75 + 
                           ue74_int + ue75_int,
                         data = dw_data,
                         num.trees = 500,
                         mtry = 3,
                         probability = TRUE,
                         min.node.size = 10,
                         importance = "impurity")

# Get propensity scores from random forest
dw_data$p_rf <- predict(rf_model, data=dw_data)$predictions[, 2]

# Trim to [0.05, 0.95]
dw_rf_trimmed <- dw_data[dw_data$p_rf >= 0.05 & dw_data$p_rf <= 0.95, ]

# Calculate weights and Y_tilde
dw_rf_trimmed$weight <- ifelse(dw_rf_trimmed$treated_int == 1,
                               1 / dw_rf_trimmed$p_rf,
                               1 / (1 - dw_rf_trimmed$p_rf))
dw_rf_trimmed$y_tilde <- dw_rf_trimmed$weight * dw_rf_trimmed$re78

# OLS with Y_tilde
rf_weighted_model <- lm(y_tilde ~ treated_int + age + education + black_int + 
                          hispanic_int + married_int + re74 + re75 + 
                          ue74_int + ue75_int,
                        data = dw_rf_trimmed)

print(summary(rf_weighted_model))
ate_rf <- mean(dw_rf_trimmed$y_tilde)
ate_rf


################################################################################
#(d) Heterogeneity in CATEs - Several approaches
################################################################################

# Re-estimate models for heterogeneity analysis
propensity_logit <- glm(treated_int ~ age + education + black_int + hispanic_int + 
                          married_int + re74 + re75 + ue74_int + ue75_int,
                        data = dw_data, family = binomial(link = "logit"))
dw_data$p_logit <- predict(propensity_logit, type = "response")
dw_logit <- dw_data[dw_data$p_logit >= 0.05 & dw_data$p_logit <= 0.95, ]
dw_logit$weight <- ifelse(dw_logit$treated_int == 1,
                          1 / dw_logit$p_logit,
                          1 / (1 - dw_logit$p_logit))
dw_logit$y_tilde <- dw_logit$weight * dw_logit$re78

# APPROACH 1: Examine coefficients on covariates in Y_tilde regressions
logit_ols <- lm(y_tilde ~ treated_int + age + education + black_int + 
                  hispanic_int + married_int + re74 + re75 + 
                  ue74_int + ue75_int, data = dw_logit)

tree_ols <- lm(y_tilde ~ treated_int + age + education + black_int + 
                 hispanic_int + married_int + re74 + re75 + 
                 ue74_int + ue75_int, data = dw_tree_trimmed)

rf_ols <- lm(y_tilde ~ treated_int + age + education + black_int + 
               hispanic_int + married_int + re74 + re75 + 
               ue74_int + ue75_int, data = dw_rf_trimmed)

cat("APPROACH 1: Coefficients on covariates\n")
print(summary(logit_ols)$coefficients)
print(summary(tree_ols)$coefficients)
print(summary(rf_ols)$coefficients)

# APPROACH 2: Interaction terms
logit_interact <- lm(y_tilde ~ treated_int * (age + education + black_int + 
                                                married_int + re74 + re75), data = dw_logit)

cat("\nAPPROACH 2: Interaction terms\n")
print(summary(logit_interact))

# Compare R² with and without interactions
r2_no_int <- summary(logit_ols)$r.squared
r2_with_int <- summary(logit_interact)$r.squared
cat("\nR² without interactions:", round(r2_no_int, 4), "\n")
cat("R² with interactions:   ", round(r2_with_int, 4), "\n")

# F-test for joint significance of interactions
anova_result <- anova(logit_ols, logit_interact)
print(anova_result)

# APPROACH 3: Tree splits - sample splitting decisions
cat("\nAPPROACH 3: Tree structure and variable importance\n")
print(tree_model)

# APPROACH 4: Random forest variable importance plot
cat("\nAPPROACH 4: Random Forest variable importance\n")
print(importance(rf_model))

# APPROACH 5: R² comparison across models
r2_logit <- summary(logit_ols)$r.squared
r2_tree <- summary(tree_ols)$r.squared
r2_rf <- summary(rf_ols)$r.squared

cat("\nAPPROACH 5: R² comparison\n")
cat(sprintf("  Logit:          %.4f\n", r2_logit))
cat(sprintf("  Tree:           %.4f\n", r2_tree))
cat(sprintf("  Random Forest:  %.4f\n", r2_rf))

################################################################################
#(e) Propensity score matching
################################################################################

# STEP (i): Nearest neighbor matching
formula <- treated_int ~ age + education + black_int + hispanic_int + 
  married_int + re74 + re75 + ue74_int + ue75_int

m.out <- matchit(formula, data = dw_data, method = "nearest")

print(summary(m.out))

# STEP (ii): Create matched data
matched_data <- match.data(m.out)

# STEP (iii): Estimate ATE using matched data
matched_model <- lm(re78 ~ treated_int + factor(subclass), 
                    data = matched_data,
                    weights = weights)

print(summary(matched_model))
ate_match <- coef(summary(matched_model))["treated_int", "Estimate"]
se_match  <- coef(summary(matched_model))["treated_int", "Std. Error"]
ate_match
se_match


################################################################################
### PROBLEM 3
################################################################################
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


################################################################################
### PROBLEM 4
################################################################################
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
