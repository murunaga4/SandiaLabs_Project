# Libraries
#install.packages("corrplot")  # Install if not already installed
#install.packages("ggcorrplot")  # Alternative package
#install.packages("data.table") 
#install.packages("corrplot")  # Install if not already installed
#install.packages("ggcorrplot")  # Alternative package
#install.packages("gbm") 
#install.packages("Metrics") 
#install.packages("MLmetrics")
#install.packages("caret") 
#install.packages("rpart.plot")
#install.packages("xgboost")
#install.packages("caret")
#install.packages("Matrix")
#install.packages("dplyr")
#install.packages("pROC")
#install.packages("Hmisc")
#install.packages("mgcv")

library(mgcv)
library(corrplot)
library(ggcorrplot)
library(data.table)  
library(pROC)
library(xgboost)
library(Matrix)
library(dplyr)
library(corrplot)
library(ggcorrplot)
library(data.table)
library(caret) 
library(rpart.plot)
library(MLmetrics)
library(corrplot)
library(RColorBrewer)
library(tidyverse)
library(ggplot2) 
#library(ggthemes) 
library(caret) 
library(randomForest)
library(gbm) 
library(rpart) 
library(Metrics) 
library(dplyr) 
library(gam) 
library(e1071) 
#library(GGally) 
library(reshape2) 
library(tidyr) 
library(corrplot) 
library(glmnet) 
library(magrittr)
library(MASS)


## Read Loan Data Set
ln_raw <- read.table("C:/Users/nisha/OneDrive/Desktop/Nisha HW/Practicum/Practicum/loan.csv", header= TRUE, sep = ",")

ln <- copy(ln_raw )


#Find unique String values
unique(ln$EducationLevel)
unique(ln$MaritalStatus)
unique(ln$EmploymentStatus)
unique(ln$LoanPurpose)
unique(ln$HomeOwnershipStatus)

# Convert String values into Numerical
ln$EducationLevel <- as.numeric(factor(ln$EducationLevel,
                                        levels = c("High School","Associate","Bachelor","Master","Doctorate"), 
                                        labels = c(1, 2, 3, 4, 5)))
ln$MaritalStatus <- as.numeric(factor(ln$MaritalStatus,
                                       levels = c( "Single","Married","Divorced","Widowed" ),
                                       labels = c(1, 2, 3, 4)))

ln$EmploymentStatus <- as.numeric(factor(ln$EmploymentStatus,
                                      levels = c("Employed","Self-Employed","Unemployed") ,
                                      labels = c(1, 2, 3)))


ln$HomeOwnershipStatus <- as.numeric(factor(ln$HomeOwnershipStatus,
                                         levels = c("Own","Mortgage","Rent","Other" ) ,
                                         labels = c(1, 2, 3,4)))

ln$LoanPurpose <- as.numeric(factor(ln$LoanPurpose,
                                         levels = c("Home","Debt Consolidation","Education","Auto","Other" ) ,
                                         labels = c(1, 2, 3,4,5)))




#head(ln[, c("RiskScore")])


head(ln[c("EducationLevel","MaritalStatus","EmploymentStatus","LoanPurpose","HomeOwnershipStatus", "RiskScore")])

#only Numberic - Remove Dates from dataset
ln_num <- ln[,-1]

# Remove Constant Variables from the dataset if any 
ln1<- ln_num[, sapply(ln_num, function(x) length(unique(x)) > 1)]

ln_new <- ln_num[c("BaseInterestRate","TotalDebtToIncomeRatio","TotalAssets", "InterestRate","MonthlyIncome","AnnualIncome","CreditScore","LoanAmount", "Experience","Age","RiskScore","LoanApproved" )]
head(ln_new)

# From VIF model , Age, Experience, Monthly Income and Annual Income are problematic with strong multicollinearity
# so remove experience and Monthly Income from the final Model
ln_new1 = ln_num[c("BaseInterestRate","TotalDebtToIncomeRatio","TotalAssets", "InterestRate","AnnualIncome","CreditScore","LoanAmount","Age","RiskScore","LoanApproved" )]
head(ln_new1)
ln_new2 = ln_num[c("BaseInterestRate","TotalDebtToIncomeRatio","TotalAssets", "InterestRate","AnnualIncome","CreditScore","LoanAmount","Age","RiskScore","LoanApproved" )]
#head(ln_new2)


ln_new2$RiskScore <- as.numeric(as.character(ln_new2$RiskScore))
ln_new2$RiskScore <- cut(ln_new2$RiskScore,
                         breaks = c(28, 35, 45, 55, 100),
                         labels = c("1", "2", "3", "4"),
                         include.lowest = TRUE,
                         right = FALSE)

ln_new2$RiskScore <- as.numeric(as.character(ln_new2$RiskScore))

head(ln_new2)


#splitting data
sample = floor(0.70 * nrow(ln_new2)) #70% of the data
set.seed(123)
sub = sample(seq_len(nrow(ln_new2)), size = sample)

cvd_train1 = ln_new2[sub,]
cvd_test1 = ln_new2[-sub,]

dim(cvd_train1)
dim(cvd_test1)


#######################################LDA################################
LDA <-lda(RiskScore ~ .,data=cvd_train1)

#LDA Training data accuracy
p1 <- predict(LDA, cvd_train1)$class

#accuracy and F1 calculation
eval_model <- function(actual,predicted) {
  conf_matrix <- confusionMatrix(as.factor(predicted), as.factor(actual))
  if (is.vector(conf_matrix$byClass)){f1_score <- conf_matrix$byClass["F1"]
  }else{
    f1_score <- mean(conf_matrix$byClass[,"F1"], na.rm = TRUE)}
  
  list(accuracy = conf_matrix$overall['Accuracy'], f1_score = f1_score)
}

ldaTrain_eval <- eval_model(cvd_train1$RiskScore, p1)
ldaTrain_eval 

#LDA Testing data accuracy
p2 <- predict(LDA, cvd_test1)$class
ldaTest_eval <- eval_model(cvd_test1$RiskScore, p2)
ldaTest_eval 


#######################################QDA################################
#ln_new2$RiskScore <- as.factor(ln_new2$RiskScore)
#QDA<-qda(RiskScore ~ .,data=cvd_train1)
#QDA Training data accuracy
#p3 <- predict(QDA, cvd_train1)$class
#qdaTrain_eval <- eval_model(cvd_train1$RiskScore, p3)
#qdaTrain_eval 


#QDA Testing data accuracy
#p4 <- predict(QDA, cvd_test1)$class
#qdaTest_eval <- eval_model(cvd_test1$RiskScore, p4)
#qdaTest_eval 

#######################################Naive Bayers################################
NB<-naiveBayes(RiskScore ~ .,data=cvd_train1)
#Naive Training data accuracy
p5 <- predict(NB, cvd_train1)
NBTrain_eval <- eval_model(cvd_train1$RiskScore, p5)
NBTrain_eval 



#Naive Testing data accuracy
p6 <- predict(NB, cvd_test1)
NBTest_eval <- eval_model(cvd_test1$RiskScore, p6)
NBTest_eval 

#######################################RainForest################################
cvd_train1$RiskScore <- as.factor(cvd_train1$RiskScore)
cvd_test1$RiskScore <- as.factor(cvd_test1$RiskScore)
#random forest with default parameters 
RF_default <- randomForest(as.factor(RiskScore)  ~ ., data = cvd_train1, importance=TRUE) 
RF_default 
importance(RF_default, type=2) 
varImpPlot(RF_default) 
RF_default_pred<-predict(RF_default,cvd_test1) 
mean(RF_default_pred!=cvd_test1$RiskScore) 
table(RF_default_pred,cvd_test1$RiskScore) 

#random forest data accuracy
p9 <- predict(RF_default, cvd_train1)
RFTrain_eval <- eval_model(cvd_train1$RiskScore, p9)
RFTrain_eval 

p10 <- predict(RF_default, cvd_test1)
RFTest_eval <- eval_model(cvd_test1$RiskScore, p10)
RFTest_eval 



#rainforest - parameter tuning  

# Random Forest 
set.seed(2024)  
# Set up a grid of parameters till 1000 
ntree_grid <- c(100, 200, 300, 400, 500, 700, 800, 1000)  
# TREE SAMPLES 
mtry_grid <- round(sqrt(ncol(cvd_train1)))  
# Tuning 
tuning_results <- expand.grid(ntree = ntree_grid, mtry = mtry_grid)  
tuning_results$error <- NA # Loop over the parameter grid  
for (i in 1:nrow(tuning_results)) {  
  # Get the parameters  
  ntree <- tuning_results$ntree[i]  
  mtry <- tuning_results$mtry[i] 
  
  RF_model <-  randomForest(RiskScore ~ .,data=cvd_train1, ntree=mtry) 
  RF_pred <- predict(RF_model,cvd_test1) 
  RF_acc <- sum(RF_pred == cvd_test1$RiskScore)/length(cvd_test1$RiskScore) 
  tuning_results$error[i] <- 1 - RF_acc 
} 
best_parm <- tuning_results[which.min(tuning_results$error),] 
print(best_parm) 
# RANDOM FOREST USING BEST PARAMeter 
RF_mod1 <- randomForest(as.factor(RiskScore)  ~ .,data=cvd_train1,  
                        ntree = 700,  
                        mtry = 3,  
                        nodesize = 6,  
                        importance=TRUE)  
## Check Varianbles  
importance(RF_mod1)  

#PLOT
importance(RF_mod1, type=2)  
varImpPlot(RF_mod1)  
#Confusion MATRIX 
RF_pred1 <- predict(RF_mod1, cvd_test1, type='class')  
RF_pred <- as.factor(RF_pred1) 
cvd_test1$RiskScore<- as.factor(cvd_test1$RiskScore) 
RF <-confusionMatrix(RF_pred, reference = cvd_test1$RiskScore) 
RF$overall[1] 

RF_pred2 <- predict(RF_mod1, cvd_train1, type='class')  
RF_pred3 <- as.factor(RF_pred2) 
cvd_train1$RiskScore<- as.factor(cvd_train1$RiskScore) 
RF <-confusionMatrix(RF_pred3, reference = cvd_train1$RiskScore) 
RF$overall[1] 


#RandomForest F1_score
F1_Score( RF_pred3, cvd_train1$RiskScore)
F1_Score( RF_pred, cvd_test1$RiskScore)

################svm###########################
#SVM default
SVM_model_default <- svm( as.factor(RiskScore)  ~ .,      
                          data=cvd_train1,          
                          kernel='linear',    
                          cost=10,            
                          scale=TRUE,
                          probability = TRUE)


#SVM default Accuracy

p10 <- predict(SVM_model_default, newdata = cvd_train1)
SVMdefTrain_eval <- mean(p10 == cvd_train1$RiskScore)
SVMdefTrain_eval 

p11 <- predict(SVM_model_default, cvd_test1)
SVMdefTest_eval <- eval_model(cvd_test1$RiskScore, p11)
SVMdefTest_eval 


library(MLmetrics)

#SVM default F1_score
#F1_Score( p10, cvd_train1$RiskScore, positive = "1")
#F1_Score( p11, cvd_test1$RiskScore, positive = "1")
#table(p10)
#table(cvd_train1$RiskScore)

#table(p11)
#table(cvd_test1$RiskScore)

confusionMatrix(p10, cvd_train1$RiskScore)
confusionMatrix(p11, cvd_test1$RiskScore)

# Training
cm_train <- confusionMatrix(p10, cvd_train1$RiskScore)
precision_train <- cm_train$byClass[, "Pos Pred Value"]
recall_train <- cm_train$byClass[, "Sensitivity"]
support_train <- as.numeric(table(cvd_train1$RiskScore))

# Testing
cm_test <- confusionMatrix(p11, cvd_test1$RiskScore)
precision_test <- cm_test$byClass[, "Pos Pred Value"]
recall_test <- cm_test$byClass[, "Sensitivity"]
support_test <- as.numeric(table(cvd_test1$RiskScore))

# Training
f1_train <- 2 * precision_train * recall_train / (precision_train + recall_train)
f1_train[is.nan(f1_train)] <- 0  # Replace NaN with 0

# Testing
f1_test <- 2 * precision_test * recall_test / (precision_test + recall_test)
f1_test[is.nan(f1_test)] <- 0

# Training
weighted_f1_train <- sum(f1_train * support_train) / sum(support_train)

# Testing
weighted_f1_test <- sum(f1_test * support_test) / sum(support_test)

# Print results
cat("Weighted F1 (Train):", round(weighted_f1_train, 4), "\n")
cat("Weighted F1 (Test):", round(weighted_f1_test, 4), "\n")



##############################GAMs##############################
library(mgcv)
# Separate predictors (exclude RiskScore)
predictors <- setdiff(names(cvd_train1), "RiskScore")

# Create numeric predictor matrix
cvd_train1_numeric <- model.matrix(~ . - 1, data = cvd_train1[, predictors])

# Combine with RiskScore
cvd_train1_clean <- data.frame(RiskScore = cvd_train1$RiskScore, cvd_train1_numeric)

# Recode RiskScore to start at 0
cvd_train1_clean$RiskScore <- as.numeric(as.character(cvd_train1_clean$RiskScore))
cvd_train1_clean$RiskScore <- cvd_train1_clean$RiskScore - min(cvd_train1_clean$RiskScore)

# Confirm values
#table(cvd_train1_clean$RiskScore) 
cvd_train1$RiskScore <- cvd_train1_clean$RiskScore + 3

#table(cvd_train1_clean$RiskScore)

base_formula <- paste("~", paste(names(cvd_train1_clean)[-1], collapse = " + "))
formula_list <- list(
  as.formula(paste("RiskScore", base_formula)),  # first formula includes response
  as.formula(base_formula),
  as.formula(base_formula)
)
GAM_model <- gam(formula_list, family = multinom(K = 3), data = cvd_train1_clean)

# Use the same predictors used in training
predictors <- names(cvd_train1_clean)[-1]  # exclude RiskScore

# Create dummy variables for test set
cvd_test1_numeric <- model.matrix(~ . - 1, data = cvd_test1[, predictors])
cvd_test1_clean <- data.frame(RiskScore = cvd_test1$RiskScore, cvd_test1_numeric)

# Recode RiskScore to match training (0 to 3)
cvd_test1_clean$RiskScore <- as.numeric(as.character(cvd_test1_clean$RiskScore))
cvd_test1_clean$RiskScore <- cvd_test1_clean$RiskScore - min(cvd_test1_clean$RiskScore)




#GAMs Accuracy

GAM1<- predict(GAM_model, cvd_train1_clean, type = "response")
# Convert to predicted class (highest probability)
GAM_pred <- apply(GAM1, 1, function(row) which.max(row) - 1)  # subtract 1 to match original coding

# Convert to factor with correct levels
GAM_pred <- factor(GAM_pred, levels = 0:3)
actual <- factor(cvd_train1_clean$RiskScore, levels = 0:3)
GAMTrain_eval <- confusionMatrix(GAM_pred, actual)
print(GAMTrain_eval)

GAM_probs_test <- predict(GAM_model, newdata = cvd_test1_clean, type = "response")
GAM_pred_test <- apply(GAM_probs_test, 1, function(row) which.max(row) - 1)
GAM_pred_test <- factor(GAM_pred_test, levels = 0:3)
actual_test <- factor(cvd_test1_clean$RiskScore, levels = 0:3)
GAMTest_eval <- confusionMatrix(GAM_pred_test, actual_test)
print(GAMTest_eval)

# Training
f1_train <- sapply(levels(actual), function(class) {
  F1_Score(y_pred = GAM_pred == class, y_true = actual == class)
})

# Testing
f1_test <- sapply(levels(actual_test), function(class) {
  F1_Score(y_pred = GAM_pred_test == class, y_true = actual_test == class)
})

# Training
support_train <- table(actual)
weighted_f1_train <- sum(f1_train * support_train) / sum(support_train)

# Testing
support_test <- table(actual_test)
weighted_f1_test <- sum(f1_test * support_test) / sum(support_test)

# Print results
cat("Weighted F1 (Train):", round(weighted_f1_train, 4), "\n")
cat("Weighted F1 (Test):", round(weighted_f1_test, 4), "\n")############Harrell's C-Statistic for LDA#########

lda_pred <- predict(LDA, cvd_test1)
lda_prob <- lda_pred$posterior[, "1"]
c_stats_lda <- roc(cvd_test1$RiskScore, lda_prob)$auc
c_stats_lda



############Harrell's C-Statistic for Naive Bayers#########
NB_pred <- predict(NB, cvd_test1, type = "raw")[, "1"] 
#NB_prob <- NB_pred$posterior[, "1"]
c_stats_NB <- roc(cvd_test1$RiskScore, NB_pred)$auc
c_stats_NB

############Harrell's C-Statistic for Random Forest Default#########
rf_prob <-  predict(RF_default, cvd_test1, type='prob')[,2]
c_stats <- roc(cvd_test1$RiskScore, rf_prob)$auc
c_stats



############Harrell's C-Statistic for Random Forest Tuned#########
rf_prob <-  predict(RF_mod1, cvd_test1, type='prob')[,2]
c_stats_tun <- roc(cvd_test1$RiskScore, rf_prob)$auc
c_stats_tun

############Harrell's C-Statistic for SVM#########
SVM_model_default <- svm(
  as.factor(RiskScore) ~ .,
  data = cvd_train1,
  kernel = "linear",
  cost = 10,
  scale = TRUE,
  probability = TRUE  
)
svm_pred <-  predict(SVM_model_default, cvd_test1, probability = TRUE)

svm_prob <- attr(svm_pred, "probabilities")
svm_prob_1 <- svm_prob[, "1"]

c_stats_svm <- roc(cvd_test1$RiskScore, svm_prob_1)$auc
c_stats_svm


##########Harrell's C-Statistic for GAMs#########
cvd_test1$RiskScore <- as.numeric(as.character(cvd_test1$RiskScore))
cvd_test1$RiskScore <- cvd_test1$RiskScore - min(cvd_test1$RiskScore)
#table(cvd_test1$RiskScore)

# Convert response to factor
actual <- factor(cvd_test1$RiskScore)
colnames(gam_prob) <- as.character(0:3)

# Compute multiclass ROC
c_stats_gam <- multiclass.roc(actual, gam_prob)

c_stats_gam

########################Predict using rainforest################################
#rainforest - deafault - train data


# RANDOM FOREST USING BEST PARAMeter 
RF_default <- randomForest(as.factor(RiskScore)  ~ ., data = cvd_train1, importance=TRUE)   

#Confusion MATRIX 
RF_pred1 <- predict(RF_default, cvd_train1, type='class')  
RF_pred <- as.factor(RF_pred1) 
RF_pred


#######################################Cross-Validation################################


B= 25; ### num of loops
TEALL = NULL; ### Final TE values
set.seed(7406); 

for (b in 1:B){
  ### randomly selecting observations
  flag <- sort(sample(1:n, size = floor(n / 5)))
  cvdtrain11 <- ln_new2[-flag,];
  cvdtest11 <- ln_new2[flag,];
  
  #Model 1- LDA 
  LDA <-lda(RiskScore ~ .,data=cvdtrain11)
  LDA_pred<-predict(LDA,cvdtest11)
  te1 <- mean(LDA_pred$class!=cvdtest11$RiskScore)
  #te1
  
  #Model 2- QDA 
  #QDA<-qda(RiskScore ~ .,data=cvdtrain11)
  #QDA_pred<-predict(QDA,cvdtest11)
  #te2 <- mean(QDA_pred$class!=cvdtest11$RiskScore) 
  #te2
  
  
  #Model 3- Naive Bayers 
  library("e1071")
  NB<-naiveBayes(RiskScore ~ .,data=cvdtrain11)
  NB_pred<-predict(NB,cvdtest11)
  te3 <- mean(NB_pred!=cvdtest11$RiskScore)
   #te3
  
  #Model 4-RandomForest
  RF_default <- randomForest(as.factor(RiskScore)  ~ ., data = cvdtrain11, importance=TRUE) 
  RF_default_pred<-predict(RF_default,cvdtest11)
  te4 <- mean(RF_default_pred!=cvdtest11$RiskScore)
  #te4
  
  #Model 5-RandomForest Tuned
  RF_tuned <- randomForest(as.factor(RiskScore)  ~ .,data=cvdtrain11, importance=TRUE, mtry=3, ntree=700, nodesize=6)
  RF_tuned_pred<-predict(RF_tuned,cvdtest11)
  te5 <- mean(RF_tuned_pred!=cvdtest11$RiskScore)
  #te5
  
  #Model 6-SVM default 
  SVM_model_default <- svm( as.factor(RiskScore)  ~ .,      
                            data=cvdtrain11,          
                            kernel='linear',    
                            cost=10,            
                            scale=TRUE)         
  
  SVM_pred_default <- predict(SVM_model_default, newdata=cvdtest11) 
  te6 <- mean(SVM_pred_default!=cvdtest11$RiskScore)
  #te6
  
  
  
  #Model 7- GAMs
  
  library(mgcv)
  predictors <- setdiff(names(cvdtrain11), "RiskScore")
  cvd_train11_numeric <- model.matrix(~ . - 1, data = cvdtrain11[, predictors])
  cvd_train11_clean <- data.frame(RiskScore = cvdtrain11$RiskScore, cvd_train11_numeric)
  cvd_train11_clean$RiskScore <- as.numeric(as.character(cvd_train11_clean$RiskScore))
  cvd_train11_clean$RiskScore <- cvd_train11_clean$RiskScore - min(cvd_train11_clean$RiskScore)
  
  cvd_train11_clean$RiskScore <- cvd_train11_clean$RiskScore - 3
  # Start fresh from original RiskScore
  cvd_train11_clean$RiskScore <- as.numeric(as.character(cvdtrain11$RiskScore))
  
  # Shift to start at 0
  cvd_train11_clean$RiskScore <- cvd_train11_clean$RiskScore - min(cvd_train11_clean$RiskScore)
  
  # Confirm values
  table(cvd_train11_clean$RiskScore)
  
  #table(cvd_train1_clean$RiskScore)
  
  base_formula <- paste("~", paste(names(cvd_train11_clean)[-1], collapse = " + "))
  formula_list <- list(
    as.formula(paste("RiskScore", base_formula)),  # first formula includes response
    as.formula(base_formula),
    as.formula(base_formula)
  )
  # Fit binary GAM model
  GAM_model <- gam(formula_list, family = multinom(K = 3), data = cvd_train11_clean)
  
  # Predict probabilities on test set
  GAM1 <- predict(GAM_model, cvdtest11, type = "response")
  GAM_probs <- predict(GAM_model, newdata = cvd_train11_clean, type = "response")
  GAM_pred <- apply(GAM_probs, 1, function(row) which.max(row) - 1)  # subtract 1 to match 0-based labels
  GAM_class <- factor(GAM_pred, levels = 0:3)
  actual_class <- factor(cvd_train11_clean$RiskScore, levels = 0:3)
  
  # Compare predictions to actual labels
  valid_idx <- !is.na(GAM_class) & !is.na(actual_class)
  te7 <- mean(GAM_class[valid_idx] != actual_class[valid_idx])
  #te7
  
  
  TEALL = rbind( TEALL, cbind(te1, te3, te4, te5, te6, te7) );
}
dim(TEALL); 
colnames(TEALL) <- c("LDA", "NaiveBayers","Random Forest Default","Random Forest Tuned","SVM", "GAMS");

apply(TEALL, 2, mean);
apply(TEALL, 2, var);

