# PS 4 
# Problem 1 

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
