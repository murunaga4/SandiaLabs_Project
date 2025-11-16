# =====================================================
# Loan Approval Prediction Dashboard - Enhanced EDA
# =====================================================

library(shiny)
library(corrplot)
library(ggcorrplot)
library(randomForest)
library(MASS)
library(e1071)
library(caret)
library(pROC)
library(MLmetrics)
library(dplyr)
library(gam)
library(ggplot2)
library(gridExtra)

# =====================================================
# Load Data & Preprocess
# =====================================================

ln_raw <- read.csv("loan.csv")

ln <- ln_raw %>%
  mutate(
    EducationLevel = as.numeric(factor(EducationLevel, 
                                       levels = c("High School","Associate","Bachelor","Master","Doctorate"), 
                                       labels = 1:5)),
    MaritalStatus = as.numeric(factor(MaritalStatus, 
                                      levels = c("Single","Married","Divorced","Widowed"), 
                                      labels = 1:4)),
    EmploymentStatus = as.numeric(factor(EmploymentStatus, 
                                         levels = c("Employed","Self-Employed","Unemployed"), 
                                         labels = 1:3)),
    HomeOwnershipStatus = as.numeric(factor(HomeOwnershipStatus, 
                                            levels = c("Own","Mortgage","Rent","Other"), 
                                            labels = 1:4)),
    LoanPurpose = as.numeric(factor(LoanPurpose, 
                                    levels = c("Home","Debt Consolidation","Education","Auto","Other"), 
                                    labels = 1:5))
  )

ln_new <- ln %>% 
  select(BaseInterestRate, TotalDebtToIncomeRatio, TotalAssets, InterestRate,
         AnnualIncome, CreditScore, LoanAmount, Age, RiskScore, NetWorth, LoanApproved)

ln_new$LoanApproved <- as.factor(ln_new$LoanApproved)

set.seed(123)
trainIndex <- createDataPartition(ln_new$LoanApproved, p = 0.7, list = FALSE)
cvd_train <- ln_new[trainIndex,]
cvd_test <- ln_new[-trainIndex,]

# =====================================================
# Train Models
# =====================================================

LDA_model <- lda(LoanApproved ~ ., data=cvd_train)
QDA_model <- qda(LoanApproved ~ ., data=cvd_train)
NB_model <- naiveBayes(LoanApproved ~ ., data=cvd_train)
RF_model <- randomForest(LoanApproved ~ ., data=cvd_train, ntree=200, mtry=3)
SVM_model <- svm(LoanApproved ~ ., data=cvd_train, kernel='linear', cost=10, probability = TRUE)
GAM_model <- gam(LoanApproved ~ ., family=binomial, data=cvd_train)

# =====================================================
# Helper: Model Evaluation
# =====================================================

eval_metrics <- function(model_name, model, test_data){
  pred <- NULL
  prob <- NULL
  
  if (model_name %in% c("LDA","QDA")) {
    p <- predict(model, test_data)
    pred <- p$class
    prob <- p$posterior[,2]
  } else if (model_name == "Naive Bayes") {
    p <- predict(model, test_data, type="raw")
    pred <- ifelse(p[, "1"] > 0.5, 1, 0)
    prob <- p[, "1"]
  } else if (model_name == "Random Forest") {
    pred <- predict(model, test_data)
    prob <- predict(model, test_data, type="prob")[,2]
  } else if (model_name == "SVM") {
    p <- attr(predict(model, test_data, probability=TRUE), "probabilities")
    prob <- p[, "1"]
    pred <- ifelse(prob > 0.5, 1, 0)
  } else if (model_name == "GAM") {
    prob <- predict(model, test_data, type="response")
    pred <- ifelse(prob > 0.5, 1, 0)
  }
  
  acc <- mean(pred == test_data$LoanApproved)
  f1 <- F1_Score(pred, test_data$LoanApproved)
  cstat <- roc(as.numeric(test_data$LoanApproved), as.numeric(prob))$auc
  
  data.frame(Model=model_name, Accuracy=round(acc,3), F1=round(f1,3), C_stat=round(cstat,3))
}

# =====================================================
# UI
# =====================================================

ui <- fluidPage(
  titlePanel("Loan Approval Prediction Dashboard"),
  
  sidebarLayout(
    sidebarPanel(
      h3("Navigation"),
      tabsetPanel(
        id="tabs", type="pills",
        tabPanel("Prediction"),
        tabPanel("Correlation Heatmap"),
        tabPanel("EDA")
      ),
      hr(),
      
      conditionalPanel(
        condition = "input.tabs=='Prediction'",
        h4("Applicant Details"),
        numericInput("BaseInterestRate", "Base Interest Rate", 0.05),
        numericInput("TotalDebtToIncomeRatio", "Debt-to-Income Ratio", 0.4),
        numericInput("TotalAssets", "Total Assets", 100000),
        numericInput("InterestRate", "Interest Rate", 0.08),
        numericInput("AnnualIncome", "Annual Income", 50000),
        numericInput("CreditScore", "Credit Score", 700),
        numericInput("LoanAmount", "Loan Amount", 20000),
        numericInput("Age", "Age", 35),
        numericInput("RiskScore", "Risk Score", 0.5),
        actionButton("predictBtn", "Predict Loan Approval"),
        selectInput("modelType", "Select Model:",
                    choices = c("LDA","QDA","Naive Bayes","Random Forest","SVM","GAM"))
      ),
      
      conditionalPanel(
        condition = "input.tabs=='EDA'",
        h4("EDA Variable"),
        selectInput("edaVar", "Variable:",
                    choices = c("Age","InterestRate","CreditScore","AnnualIncome","RiskScore","NetWorth"))
      )
    ),
    
    mainPanel(
      conditionalPanel(
        condition = "input.tabs=='Prediction'",
        h2("Prediction Result"),
        textOutput("predictionResult"),
        hr(),
        h3("Model Performance Metrics"),
        tableOutput("modelMetrics")
      ),
      conditionalPanel(
        condition = "input.tabs=='Correlation Heatmap'",
        h3("Correlation Heatmap"),
        plotOutput("corrPlot", height="500px")
      ),
      conditionalPanel(
        condition = "input.tabs=='EDA'",
        h3("Exploratory Data Analysis"),
        plotOutput("edaBoxHist", height="400px"),
        hr(),
        h4("Summary Statistics"),
        tableOutput("edaStats"),
        hr(),
        h4("Statistical Analysis"),
        textOutput("edaSummary")
      )
    )
  )
)

# =====================================================
# Server
# =====================================================

server <- function(input, output) {
  
  models <- list(
    "LDA"=LDA_model, "QDA"=QDA_model, "Naive Bayes"=NB_model,
    "Random Forest"=RF_model, "SVM"=SVM_model, "GAM"=GAM_model
  )
  
  observeEvent(input$predictBtn, {
    newdata <- data.frame(
      BaseInterestRate=input$BaseInterestRate,
      TotalDebtToIncomeRatio=input$TotalDebtToIncomeRatio,
      TotalAssets=input$TotalAssets,
      InterestRate=input$InterestRate,
      AnnualIncome=input$AnnualIncome,
      CreditScore=input$CreditScore,
      LoanAmount=input$LoanAmount,
      Age=input$Age,
      RiskScore=input$RiskScore,
      NetWorth=mean(cvd_train$NetWorth, na.rm=TRUE)
    )
    
    model <- models[[input$modelType]]
    
    if (input$modelType %in% c("LDA","QDA")) {
      p <- predict(model, newdata)
      pred_class <- p$class
      prob <- p$posterior[, "1"]
    } else if (input$modelType=="Naive Bayes") {
      p <- predict(model, newdata, type="raw")
      pred_class <- ifelse(p[, "1"]>0.5,1,0)
      prob <- p[, "1"]
    } else if (input$modelType=="Random Forest") {
      p <- predict(model, newdata, type="prob")
      pred_class <- ifelse(p[, "1"]>0.5,1,0)
      prob <- p[, "1"]
    } else if (input$modelType=="SVM") {
      p <- attr(predict(model, newdata, probability=TRUE),"probabilities")
      prob <- p[, "1"]
      pred_class <- ifelse(prob>0.5,1,0)
    } else if (input$modelType=="GAM") {
      prob <- predict(model, newdata, type="response")
      pred_class <- ifelse(prob>0.5,1,0)
    }
    
    pred_label <- ifelse(pred_class==1,"✅ Loan Approved","🚫 Loan Not Approved")
    output$predictionResult <- renderText({
      paste(pred_label, "(Probability:", round(prob,3),")")
    })
    
    metrics <- eval_metrics(input$modelType, model, cvd_test)
    output$modelMetrics <- renderTable(metrics)
  })
  
  # Correlation
  output$corrPlot <- renderPlot({
    cor_matrix <- cor(ln_new %>% select(-LoanApproved), use="complete.obs")
    ggcorrplot(cor_matrix, type="lower", hc.order=TRUE, lab=TRUE, lab_size=2,
               colors=c("blue","white","red"))
  })
  
  # EDA Boxplot + Histogram
  output$edaBoxHist <- renderPlot({
    var <- input$edaVar
    x <- ln[[var]]
    if(!is.numeric(x)) return(NULL)
    
    boxp <- ggplot(ln, aes_string(y=var)) + 
      geom_boxplot(fill="steelblue", outlier.colour="red", outlier.shape=8) +
      labs(title=paste("Boxplot of", var), y=var) +
      theme_minimal()
    
    histp <- ggplot(ln, aes_string(x=var)) +
      geom_histogram(fill="orange", bins=30, color="black", alpha=0.7) +
      labs(title=paste("Histogram of", var), x=var, y="Count") +
      theme_minimal()
    
    gridExtra::grid.arrange(boxp, histp, nrow=1)
  })
  
  # EDA Stats Table
  output$edaStats <- renderTable({
    var <- input$edaVar
    x <- ln[[var]]
    if(!is.numeric(x)) return(NULL)
    data.frame(
      Variable=var,
      Mean=mean(x, na.rm=TRUE),
      Median=median(x, na.rm=TRUE),
      SD=sd(x, na.rm=TRUE),
      Min=min(x, na.rm=TRUE),
      Max=max(x, na.rm=TRUE),
      Q1=quantile(x,0.25,na.rm=TRUE),
      Q3=quantile(x,0.75,na.rm=TRUE)
    )
  })
  
  # EDA Textual Summary
  output$edaSummary <- renderText({
    var <- input$edaVar
    x <- ln[[var]]
    if(!is.numeric(x)) return(NULL)
    
    skw <- e1071::skewness(x, na.rm=TRUE)
    skew_text <- if(skw>0.5) "right-skewed" else if(skw< -0.5) "left-skewed" else "approximately symmetric"
    
    Q1 <- quantile(x,0.25,na.rm=TRUE)
    Q3 <- quantile(x,0.75,na.rm=TRUE)
    IQR <- Q3 - Q1
    outliers <- sum(x < (Q1-1.5*IQR) | x > (Q3+1.5*IQR), na.rm=TRUE)
    outlier_text <- if(outliers>0) paste(outliers,"outliers detected") else "no outliers detected"
    
    paste("Variable", var, "is", skew_text, "with", outlier_text, ".")
  })
}

# =====================================================
# Run App
# =====================================================

shinyApp(ui=ui, server=server)
