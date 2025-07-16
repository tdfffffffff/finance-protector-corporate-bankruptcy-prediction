setwd("C:/Users/user/Downloads/BC2407 Course Materials/BC2407 Course Materials/project")

# Load libraries
library(caret)
library(smotefamily)
library(xgboost)
library(randomForest)
library(pROC)
library(glmnet)
library(FNN)
library(dplyr)
library(tidyr)
library(forcats)
library(scales)
library(tidyverse)
library(ggplot2)
library(ggridges)
library(GGally)
library(corrplot)
library(MLmetrics)
library(vip)
library(gridExtra)
library(broom)
library(stringr)

# Load dataset
df <- read.csv("data.csv")  # Choose 'data 2.csv' or your actual dataset
df$Bankrupt. <- as.factor(df$Bankrupt.)  # Make target a factor

#EDA
# SMD
df <- read.csv("data.csv")

# Split data
df_0 <- df %>% filter(Bankrupt. == 0)
df_1 <- df %>% filter(Bankrupt. == 1)

# Compute SMD
calc_smd <- function(var_name) {
  mean_0 <- mean(df_0[[var_name]])
  mean_1 <- mean(df_1[[var_name]])
  sd_0 <- sd(df_0[[var_name]])
  sd_1 <- sd(df_1[[var_name]])
  pooled_sd <- sqrt((sd_0^2 + sd_1^2) / 2)
  smd <- (mean_1 - mean_0) / pooled_sd
  return(smd)
}

# Run on all numeric features
feature_names <- names(df)[names(df) != "Bankrupt."]
smd_values <- sapply(feature_names, calc_smd)

# Prepare dataframe
smd_df <- data.frame(
  Feature = names(smd_values),
  SMD = smd_values
) %>%
  arrange(desc(abs(SMD))) %>%
  slice(1:10) %>%
  mutate(
    Direction = ifelse(SMD > 0, "Higher in Bankrupt Firms", "Higher in Non-Bankrupt Firms"),
    Feature = fct_reorder(Feature, abs(SMD))
  )

# Plot
ggplot(smd_df, aes(x = Feature, y = SMD, fill = Direction)) +
  geom_col() +
  coord_flip() +
  geom_text(aes(label = round(SMD, 2)), hjust = ifelse(smd_df$SMD > 0, -0.1, 1.1), size = 3.5) +
  scale_fill_manual(values = c("Higher in Bankrupt Firms" = "#D7263D", "Higher in Non-Bankrupt Firms" = "#1B998B")) +
  labs(
    title = "Top 10 Features by Standardized Mean Difference (SMD)",
    subtitle = "Colored by which group had the higher mean",
    x = "Feature",
    y = "Standardized Mean Difference",
    fill = "Group with Higher Mean"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 11),
    axis.text.y = element_text(size = 10)
  ) +
  expand_limits(y = c(-max(abs(smd_df$SMD)) * 1.4, max(abs(smd_df$SMD)) * 1.4))

#Density Plot

top5_features <- names(sort(abs(smd_values), decreasing = TRUE))[1:5]
df_long <- df %>%
  select(Bankrupt., all_of(top5_features)) %>%
  pivot_longer(-Bankrupt., names_to = "Feature", values_to = "Value")

ggplot(df_long, aes(x = Value, fill = as.factor(Bankrupt.))) +
  geom_density(alpha = 0.4) +
  facet_wrap(~ Feature, scales = "free", ncol = 2) +
  scale_fill_manual(values = c("0" = "#1B998B", "1" = "#D7263D")) +
  labs(title = "Density Plots of Top Features by Bankruptcy Status", fill = "Bankrupt?") +
  theme_minimal()

# individual density plots
df <- read.csv("data.csv")
df$Bankrupt. <- as.factor(df$Bankrupt.)

# Set consistent theme
theme_set(theme_minimal())

ggplot(df, aes(x = ROA.A..before.interest.and...after.tax, fill = Bankrupt.)) +
  geom_density(alpha = 0.6) +
  geom_vline(data = df %>% filter(Bankrupt. == 1), 
             aes(xintercept = median(ROA.A..before.interest.and...after.tax)),
             linetype = "dashed", color = "red") +
  labs(title = "ROA vs Bankruptcy: Some Bankrupt Firms Still Look Profitable", 
       x = "ROA (A)", y = "Density")

ggplot(df, aes(x = Debt.ratio.., fill = Bankrupt.)) +
  geom_density(alpha = 0.6) +
  labs(title = "Debt Ratio Distribution: Bankrupt vs Non-Bankrupt Firms", x = "Debt Ratio")

ggplot(df, aes(x = Bankrupt., y = Borrowing.dependency, fill = Bankrupt.)) +
  geom_boxplot() +
  labs(title = "Borrowing Dependency by Bankruptcy", x = "Bankruptcy", y = "Borrowing Dependency")


# Train-test split
set.seed(123)
split <- createDataPartition(df$Bankrupt., p = 0.7, list = FALSE)
train_df <- df[split, ]
test_df <- df[-split, ]

# Apply SMOTE only on train set, test on real world data
train_x <- train_df[, !colnames(train_df) %in% "Bankrupt."]
train_y <- as.numeric(as.character(train_df$Bankrupt.))  # Convert to numeric (0/1)

smote_result <- SMOTE(X = train_x, target = train_y, K = 5)
train_bal <- data.frame(smote_result$data)
colnames(train_bal)[ncol(train_bal)] <- "Bankrupt"

# data cleaning
smote_df <- data.frame(
  smote_result$data
)

# remove noise

# Calculate average distance to k-nearest real neighbors
k <- 5
real_minority <- as.matrix(train_x[train_y == 1, ])
synthetic_samples <- smote_df[!rownames(smote_df) %in% rownames(train_x), -ncol(smote_df)]
synthetic_matrix <- as.matrix(synthetic_samples)

k <- 5
nn_dist <- knnx.dist(
  data = real_minority,
  query = synthetic_matrix,
  k = k
)

noise_threshold <- quantile(nn_dist[,k], 0.95)
noisy_samples <- which(nn_dist[,k] > noise_threshold)

# Remove noisy synthetic samples
clean_smote <- smote_df[-noisy_samples,]

# write.csv(clean_smote, "SMOTE_data.csv", row.names = FALSE) #running this should get the SMOTE_data.csv
# write.csv(test_df, "test_data.csv", row.names = FALSE)

# Modelling
#MARS
train <- read.csv("train.csv")  

# Fit a basic MARS model
mars1 <- earth(
  class ~ .,       # class is our target column
  data = train     # using training data
)

# Print model summary
print(mars1)

summary(mars1)

# Check model coefficients
summary(mars1)$coefficients %>% head(10)

# Plot model to inspect term selection
plot(mars1, which = 1)

# Fit a model with interactions (degree = 2)
mars2 <- earth(
  class ~ .,
  data = train,
  degree = 2
)

summary(mars2)

# View interaction term coefficients
summary(mars2)$coefficients %>% head(10)

######Tuning
# --- Ensure target column is a factor (binary classification) ---
train$class <- as.factor(train$class)

# --- Create tuning grid (degree: 1–3, nprune: 10 values from 2 to 100) ---
hyper_grid <- expand.grid(
  degree = 1:3,
  nprune = floor(seq(2, 100, length.out = 10))
)

head(hyper_grid)

# --- Rename factor levels to be valid for classification (caret expects names like "yes"/"no") ---
levels(train$class) <- c("no", "yes")  # e.g., 0 → "no", 1 → "yes"

customSummary <- function(data, lev = NULL, model = NULL) {
  precision_val <- Precision(y_pred = data$pred, y_true = data$obs, positive = "yes")
  recall_val <- Recall(y_pred = data$pred, y_true = data$obs, positive = "yes")
  
  # Avoid division by zero
  if (precision_val + recall_val == 0) {
    f1_val <- 0
  } else {
    f1_val <- 2 * precision_val * recall_val / (precision_val + recall_val)
  }
  
  return(c(F1 = f1_val))
}

# --- Define 10-fold cross-validation procedure ---
ctrl <- trainControl(
  method = "cv",
  number = 5,
  summaryFunction = customSummary,
  classProbs = FALSE,
  savePredictions = "final"
)

# --- Fit tuned MARS model using caret + earth ---
set.seed(123)
cv_mars <- train(
  x = subset(train, select = -class),
  y = train$class,
  method = "earth",
  metric = "F1",             # using F1 as the evaluation metric
  trControl = ctrl,
  tuneGrid = hyper_grid
)


# --- View best model parameters ---
print(cv_mars$bestTune)

# --- View performance of best model ---
cv_mars$results %>%
  filter(nprune == cv_mars$bestTune$nprune,
         degree == cv_mars$bestTune$degree)

# --- Plot cross-validation results ---
ggplot(cv_mars)

# --- Inspect fold-wise performance ---
cv_mars$resample

# Generate variable importance plots
p1 <- vip(cv_mars, num_features = 40, geom = "point", value = "gcv") + ggtitle("GCV")
p2 <- vip(cv_mars, num_features = 40, geom = "point", value = "rss") + ggtitle("RSS")

# Display side by side
grid.arrange(p1, p2, ncol = 2)

# Extract and print all hinge and interaction terms with their coefficients
cv_mars$finalModel %>%
  coef() %>%                      # Extract coefficients from the final MARS model
  broom::tidy() %>%              # Convert to a tidy data frame
  filter(str_detect(names, "h\\(")) %>%  # Keep only hinge terms (including interactions)
  print(n = Inf)                 # Print all rows



# Fit MARS model with interaction terms (degree=2)
mars <- earth(class ~ ., data = train, degree = 2, trace = 3)

model_summary <- summary(mars)
all_terms <- names(coef(mars))
coef_values <- coef(mars)

results_df <- data.frame(
  Term = all_terms,
  Coefficient = coef_values,
  Is_Interaction = grepl("\\*", all_terms),
  Is_Hinge = grepl("h\\(", all_terms),
  stringsAsFactors = FALSE
) %>%
  mutate(
    Term_Type = case_when(
      Is_Interaction & Is_Hinge ~ "Hinge Interaction",
      Is_Interaction ~ "Interaction",
      Is_Hinge ~ "Hinge Function",
      TRUE ~ "Linear"
    )
  ) %>%
  select(-Is_Interaction, -Is_Hinge) %>%
  arrange(desc(abs(Coefficient)))

named_coefficients <- setNames(results_df$Coefficient, results_df$Term)

library(pdp)
library(ggplot2)

# Define prediction function for probability of "yes"
predict_function <- function(object, newdata) {
  predict(object, newdata = newdata, type = "response")[, "yes"]
}

# Force correct partial function and run
partial <- pdp::partial

# Generate the PDP
pdp_obj <- partial(
  .f = predict_function,
  object = cv_mars$finalModel,
  pred.var = "ROA.A..before.interest.and...after.tax",
  train = train,
  grid.resolution = 10,
  progress = "text"
)

print(class(pdp_obj))  # Should be "partial" and "data.frame"
head(pdp_obj)          # Should show columns like: variable and yhat

ggplot(data = pdp_obj, aes(x = ROA.A..before.interest.and...after.tax, y = yhat)) +
  geom_line(color = "steelblue", size = 1.2) +
  labs(
    title = "Partial Dependence Plot",
    x = "ROA (Before Interest and After Tax)",
    y = "Predicted Probability of Bankruptcy"
  ) +
  theme_minimal(base_size = 14)


# PDP for "Net.worth.Assets"
pdp_networth <- partial(
  .f = predict_function,
  object = cv_mars$finalModel,
  pred.var = "Net.worth.Assets",
  train = train,
  grid.resolution = 10,
  progress = "text"
)

# Plot
ggplot(data = pdp_networth, aes(x = Net.worth.Assets, y = yhat)) +
  geom_line(color = "seagreen", size = 1.2) +
  labs(
    title = "Partial Dependence Plot",
    x = "Net Worth / Assets",
    y = "Predicted Probability of Bankruptcy"
  ) +
  theme_minimal(base_size = 14)

# Convert first two into ggplot objects
plot1 <- ggplot(data = pdp_obj, aes(x = ROA.A..before.interest.and...after.tax, y = yhat)) +
  geom_line(color = "steelblue", size = 1.2) +
  labs(title = "PDP: ROA (Before Interest and After Tax)", x = "ROA", y = "Predicted Probability") +
  theme_minimal(base_size = 12)

plot2 <- ggplot(data = pdp_networth, aes(x = Net.worth.Assets, y = yhat)) +
  geom_line(color = "seagreen", size = 1.2) +
  labs(title = "PDP: Net Worth / Assets", x = "Net Worth / Assets", y = "Predicted Probability") +
  theme_minimal(base_size = 12)

# Display side by side 
grid.arrange(plot1, plot2, ncol = 2)

# 2D PDP for interaction between ROA and Net Worth
pdp_interaction <- partial(
  .f = predict_function,
  object = cv_mars$finalModel,
  pred.var = c("ROA.A..before.interest.and...after.tax", "Net.worth.Assets"),
  train = train,
  grid.resolution = 10,
  progress = "text"
)

# 3D surface plot 
plotPartial(
  pdp_interaction,
  levelplot = FALSE,
  drape = TRUE,
  colorkey = TRUE,
  zlab = "Predicted Probability",
  screen = list(z = -20, x = -60),
  main = "Interaction PDP: ROA vs Net Worth / Assets"
)

# --- Comparing Degree 1, 2 and 3 ---
# --- Load training and test data ---
train <- read.csv("train.csv")
test_data <- read.csv("test_data.csv")
# Ensure proper format
train$class <- factor(train$class, levels = c(0,1), labels = c("no", "yes"))
test_data$Bankrupt. <- factor(test_data$Bankrupt., levels = c(0,1), labels = c("no", "yes"))

# Get common features
common_features <- intersect(names(train), names(test_data))
common_features <- setdiff(common_features, c("class", "Bankrupt."))


set.seed(123)

mars_deg1 <- earth(class ~ ., data = train, degree = 1)
mars_deg2 <- earth(class ~ ., data = train, degree = 2)
mars_deg3 <- earth(class ~ ., data = train, degree = 3)

evaluate_mars <- function(model, test_data, true_label = "Bankrupt.", pos_class = "yes") {
  # --- Input Validation ---
  if (!true_label %in% names(test_data)) {
    stop(paste("Column", true_label, "not found in test_data"))
  }
  
  # Get Predictions ---
  pred_class <- predict(model, newdata = test_data, type = "class")
  pred_prob <- predict(model, newdata = test_data, type = "response")[, pos_class]
  
  # Convert to Consistent Factors ---
  y_true <- factor(test_data[[true_label]], levels = c("no", "yes"))
  y_pred <- factor(pred_class, levels = c("no", "yes"))
  
  # Compute Confusion Matrix ---
  cm <- caret::confusionMatrix(y_pred, y_true, positive = "yes")
  
  # Calculate Metrics ---
  metrics <- list(
    "Confusion Matrix" = cm$table,
    "Balanced Accuracy" = cm$byClass["Balanced Accuracy"],
    "Accuracy" = cm$overall["Accuracy"],
    "Precision" = cm$byClass["Precision"],
    "Recall (Sensitivity)" = cm$byClass["Recall"],
    "Specificity" = cm$byClass["Specificity"],
    "F1 Score" = cm$byClass["F1"],
    "AUC-ROC" = pROC::auc(pROC::roc(response = y_true, predictor = pred_prob))
  )
  
  # Print Summary ---
  cat("\n--- MARS Model Evaluation ---\n")
  cat(sprintf("Balanced Accuracy : %.4f\n", metrics$`Balanced Accuracy`))
  cat(sprintf("Accuracy          : %.4f\n", metrics$Accuracy))
  cat(sprintf("Precision         : %.4f\n", metrics$Precision))
  cat(sprintf("Recall (Sensitivity): %.4f\n", metrics$`Recall (Sensitivity)`))
  cat(sprintf("Specificity       : %.4f\n", metrics$Specificity))
  cat(sprintf("F1 Score          : %.4f\n", metrics$`F1 Score`))
  cat(sprintf("AUC-ROC           : %.4f\n", metrics$`AUC-ROC`))
  
  return(metrics)
}


cat("\n>>> DEGREE 1 <<<")
res1 <- evaluate_mars(mars_deg1, test_data)

cat("\n>>> DEGREE 2 <<<")
res2 <- evaluate_mars(mars_deg2, test_data)

cat("\n>>> DEGREE 3 <<<")
res3 <- evaluate_mars(mars_deg3, test_data)

# Optional: rename test label column if needed
test_data$Bankrupt. <- as.factor(test_data$Bankrupt.)
levels(test_data$Bankrupt.) <- c("no", "yes")

# Get common features
common_features <- intersect(names(train), names(test_data))
common_features <- setdiff(common_features, c("class", "Bankrupt."))

# --- Define tuning grid ---
hyper_grid <- expand.grid(
  degree = 1:3,
  nprune = floor(seq(2, 100, length.out = 10))
)

library(MLmetrics)

customSummary <- function(data, lev = NULL, model = NULL) {
  precision_val <- Precision(y_pred = data$pred, y_true = data$obs, positive = "yes")
  recall_val <- Recall(y_pred = data$pred, y_true = data$obs, positive = "yes")
  
  # Avoid division by zero
  if (precision_val + recall_val == 0) {
    f1_val <- 0
  } else {
    f1_val <- 2 * precision_val * recall_val / (precision_val + recall_val)
  }
  
  return(c(F1 = f1_val))
}

# --- Define 10-fold cross-validation procedure ---
ctrl <- trainControl(
  method = "cv",
  number = 5,
  summaryFunction = customSummary,
  classProbs = FALSE,
  savePredictions = "final"
)

# --- Fit tuned MARS model using caret + earth ---
# might not get identical results from tuning
set.seed(123)
cv_mars <- train(
  x = subset(train, select = -class),
  y = train$class,
  method = "earth",
  metric = "F1",             # using F1 as the evaluation metric
  trControl = ctrl,
  tuneGrid = hyper_grid
)

cv_mars$results %>%
  arrange(desc(F1)) %>%
  head()

# --- 6. Check best model ---
cv_mars$bestTune

# --- 7. Predict on test set ---
mars_preds <- predict(cv_mars, newdata = test_data[, common_features])

# Create confusion matrix object
cm <- confusionMatrix(mars_preds, test_data$Bankrupt., positive = "yes")

# Extract table from confusion matrix
cm_table <- as.data.frame(cm$table)

# Rename columns for clarity
colnames(cm_table) <- c("Predicted", "Actual", "Freq")

# Plot using ggplot
ggplot(data = cm_table, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile(color = "black") +
  geom_text(aes(label = Freq), size = 6) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  theme_minimal(base_size = 14) +
  labs(title = "Confusion Matrix", fill = "Count") +
  coord_fixed()


# --- 8. Visualize performance across terms and degrees ---
ggplot(cv_mars)


#####Evaultion

library(MLmetrics)


#  Prepare test labels and predictions ---
y_test <- test_data$Bankrupt.
pred_class <- mars_preds
pred_prob <- predict(cv_mars, newdata = test_data[, common_features], type = "prob")[, "yes"]

# Ensure the labels are characters (MLmetrics needs "0"/"1" or "no"/"yes") ---
y_test <- as.character(y_test)
pred_class <- as.character(pred_class)

#  Compute metrics ---
# Extract TP, TN, P, N for Balanced Accuracy
tp <- cm$table["yes", "yes"]
tn <- cm$table["no", "no"]
p <- sum(cm$table[, "yes"])  # Total actual positives
n <- sum(cm$table[, "no"])   # Total actual negatives

# Compute Balanced Accuracy
balanced_accuracy <- 0.5 * ((tp / p) + (tn / n))

precision <- Precision(y_pred = pred_class, y_true = y_test, positive = "yes")
sensitivity    <- Recall(y_pred = pred_class, y_true = y_test, positive = "yes")
f1        <- F1_Score(y_pred = pred_class, y_true = y_test, positive = "yes")

#  AUC using pROC ---
roc_obj <- roc(y_test, as.numeric(pred_prob))
auc <- auc(roc_obj)

#  Print Results ---
cat("\n--- MARS MODEL RESULTS ---\n")
cat(sprintf("Accuracy  : %.4f\n", balanced_accuracy))
cat(sprintf("Precision : %.4f\n", precision))
cat(sprintf("Sensitvity: %.4f\n", sensitivity))
cat(sprintf("F1 Score  : %.4f\n", f1))
cat(sprintf("AUC-ROC   : %.4f\n", auc))

# RF
# Load cleaned datasets
train_df <- read.csv("train.csv")       
test_df <- read.csv("test_data.csv")

# Convert target variables to factors
train_df$class <- as.factor(train_df$class)
test_df$Bankrupt. <- as.factor(test_df$Bankrupt.)

# Align column names (match features between train and test)
common_features <- intersect(colnames(train_df), colnames(test_df))
common_features <- setdiff(common_features, c("class", "Bankrupt."))

# Prepare for caret
set.seed(123)
train_df$class <- factor(train_df$class, levels = c(0, 1))
test_df$Bankrupt. <- factor(test_df$Bankrupt., levels = c(0, 1))

# Rename levels to avoid numeric-class errors in caret
train_df$Bankrupt. <- factor(paste0("X", train_df$class))
test_df$Bankrupt. <- factor(paste0("X", test_df$Bankrupt.))

# TrainControl and Tuning Grid
control <- trainControl(method = "repeatedcv",
                        number = 5,
                        repeats = 3,
                        classProbs = TRUE,
                        summaryFunction = twoClassSummary,
                        verboseIter = TRUE)

tune_grid <- expand.grid(mtry = c(2, 5, 10, 15, 20))

# Train model with tuning
rf_tuned <- train(x = train_df[, common_features],
                  y = train_df$Bankrupt.,
                  method = "rf",
                  metric = "ROC",
                  trControl = control,
                  tuneGrid = tune_grid,
                  ntree = 500)

print(rf_tuned)
plot(rf_tuned)

#  Predict probabilities on test set
rf_probs <- predict(rf_tuned, newdata = test_df[, common_features], type = "prob")[, "X1"]

# Apply threshold to convert probabilities to class labels
threshold <- 0.249# 👈 adjust this to tune sensitivity/specificity
rf_preds <- ifelse(rf_probs > threshold, "X1", "X0")
rf_preds <- factor(rf_preds, levels = c("X0", "X1"))

# Evaluate model
conf_matrix <- confusionMatrix(rf_preds, test_df$Bankrupt., positive = "X1")
print(conf_matrix)

# Balanced Accuracy
bal_acc <- mean(c(conf_matrix$byClass["Sensitivity"], conf_matrix$byClass["Specificity"]))
cat("Balanced Accuracy (threshold =", threshold, "):", round(bal_acc, 4), "\n")

auc_score <- pROC::auc(roc_obj)

# Make sure probabilities are numeric
rf_probs <- as.numeric(rf_probs)
true_labels <- as.numeric(test_df$Bankrupt.) - 1  # "X0"/"X1" to 0/1

# Confirm no NAs
if (any(is.na(rf_probs)) || any(is.na(true_labels))) {
  stop("Error: NA values found in predictions or labels.")
}

# Compute ROC
roc_obj <- pROC::roc(response = true_labels, predictor = rf_probs, levels = c(0, 1), direction = "<")

# Compute AUC
auc_score <- pROC::auc(roc_obj)
cat("AUC:", round(auc_score, 4), "\n")

# Plot ROC
plot(roc_obj, col = "blue", main = "ROC Curve - Random Forest", legacy.axes = TRUE)
abline(a = 0, b = 1, lty = 2, col = "gray")

# F1 Score
f1 <- F1_Score(y_true = test_df$Bankrupt., y_pred = rf_preds, positive = "X1")
cat("F1 Score:", round(f1, 4), "\n")