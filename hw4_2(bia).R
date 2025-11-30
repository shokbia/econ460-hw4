library(tidyverse)
library(rpart)      
library(rpart.plot)   
library(randomForest) 
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
tree_model <- rpart(treated_int ~ age + education + black_int + hispanic_int + 
                      married_int + re74 + re75 + ue74_int + ue75_int,
                    data = dw_data,
                    method = "class",
                    control = rpart.control(cp = 0.01, minsplit = 20))

# Get propensity scores from tree
dw_data$p_tree <- predict(tree_model, type = "prob")[, 2]

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

# PART (ii): RANDOM FORESTS
dw_data$treated_factor <- as.factor(dw_data$treated_int)

set.seed(123)
rf_model <- randomForest(treated_factor ~ age + education + black_int + 
                           hispanic_int + married_int + re74 + re75 + 
                           ue74_int + ue75_int,
                         data = dw_data,
                         ntree = 500,
                         mtry = 3,
                         importance = TRUE,
                         nodesize = 10)

# Get propensity scores from random forest
dw_data$p_rf <- predict(rf_model, type = "prob")[, 2]

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
print(tree_model$variable.importance)

# APPROACH 4: Random forest variable importance plot
cat("\nAPPROACH 4: Random Forest variable importance\n")
print(importance(rf_model))
varImpPlot(rf_model, main = "Variable Importance: Random Forest")

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

m.out <- matchit(formula, 
                 data = dw_data, 
                 method = "nearest")

print(summary(m.out))

# STEP (ii): Create matched data
matched_data <- match.data(m.out)

# STEP (iii): Estimate ATE using matched data
matched_model <- lm(re78 ~ treated_int + factor(subclass), 
                    data = matched_data,
                    weights = weights)

print(summary(matched_model))
