# HW 4 
# Question 2 
# Lara Li 

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

