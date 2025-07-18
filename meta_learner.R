# ======================= Soil Spectroscopy Meta-Learner Focused Pipeline =======================

# ---- 1. Load Libraries ----
library(data.table)
library(dplyr)
library(tidyr)
library(caret)
library(prospectr)
library(glmnet)
library(quantregForest)
library(ggplot2)
library(Metrics)
library(DescTools)
library(solitude)

# ---- Timestamp ----
timestamp <- format(Sys.time(), "%Y%m%d_%H%M")

# ---- 2. Load & Clean Data ----
data <- fread("data/merged_nocf_data.csv")
wl_cols <- grep("^X[0-9]+$", names(data), value = TRUE)
wl <- as.numeric(sub("X", "", wl_cols))

# Remove OM outliers
Q1 <- quantile(data$OM, 0.25)
Q3 <- quantile(data$OM, 0.75)
IQR_OM <- Q3 - Q1
data <- data[data$OM >= (Q1 - 1.5 * IQR_OM) & data$OM <= (Q3 + 1.5 * IQR_OM), ]
y <- log1p(data$OM)

# ---- 3. Preprocess Spectra ----
keep <- wl >= 500 & wl <= 2450
X <- data[, ..wl_cols][, keep, with = FALSE]
wl <- wl[keep]
new_wl <- seq(min(wl), max(wl), by = 5)
X_resampled <- resample(X, wav = wl, new.wav = new_wl)
colnames(X_resampled) <- paste0("W", new_wl)

# ---- 4. Split & Transform Data ----
set.seed(42)
split <- createDataPartition(y, p = 0.7, list = FALSE)
X_train_raw <- X_resampled[split, ]
X_test_raw <- X_resampled[-split, ]
y_train <- y[split]
y_test <- y[-split]

# ---- 5. Spectral Preprocessing Chain ----
preprocess_chain <- function(mat) {
    mat <- log10(1 / (mat + 1e-8))
    mat <- savitzkyGolay(mat, p = 2, w = 11, m = 0)
    mat <- gapDer(mat, m = 1, w = 11, s = 3)
    return(mat)
}

X_train_deriv <- preprocess_chain(X_train_raw)
X_test_deriv <- preprocess_chain(X_test_raw)

non_zero <- apply(X_train_deriv, 2, function(x) is.finite(sd(x)) && sd(x) > 1e-10)
X_train_deriv <- X_train_deriv[, non_zero]
X_test_deriv <- X_test_deriv[, non_zero]

# ---- 6. QRF Variable Importance ----
qrf_raw <- quantregForest(X_train_deriv, y_train, ntrees = 500, importance = TRUE)
importance_df <- data.frame(
    Wavelength = as.numeric(gsub("W", "", colnames(X_train_deriv))),
    Importance = as.numeric(importance(qrf_raw))
) %>% mutate(NormImportance = Importance / sum(Importance))

soil_features <- data.frame(
    Wavelength = c(1400, 1900, 2200),
    Label = c("OM-H₂O", "Clay-OH", "Carbonate/OM")
)

top_waves <- importance_df %>% slice_max(NormImportance, n = 10)

contrib_plot <- ggplot(importance_df, aes(x = Wavelength, y = NormImportance)) +
    geom_col(fill = "red", alpha = 0.8, width = 5) +
    geom_vline(data = soil_features, aes(xintercept = Wavelength), color = "blue", linetype = "dashed") +
    geom_text(data = soil_features, aes(x = Wavelength, y = max(importance_df$NormImportance) * 0.95, label = Label),
              angle = 90, vjust = -0.5, hjust = 0, size = 3, color = "blue") +
    geom_point(data = top_waves, aes(x = Wavelength, y = NormImportance), color = "black", size = 2.5) +
    geom_text(data = top_waves, aes(x = Wavelength, y = NormImportance, label = round(Wavelength)),
              vjust = -1, size = 2.5, color = "black") +
    labs(title = "Wavelength Contribution to OM Prediction (QRF)",
         x = "Wavelength (nm)", y = "Absolute contribution") +
    theme_minimal() +
    coord_cartesian(ylim = c(0, max(importance_df$NormImportance) * 1.1))

ggsave(paste0("outputs/plots/qrf_wavelength_contribution_", timestamp, ".png"), contrib_plot, width = 10, height = 5)

# ---- 7. PCA + Outlier Removal ----
pca_model <- prcomp(X_train_deriv, center = TRUE, scale. = TRUE)
var_exp <- cumsum(pca_model$sdev^2) / sum(pca_model$sdev^2)
n_comp <- which(var_exp >= 0.99)[1]
X_train_pca <- predict(pca_model, X_train_deriv)[, 1:n_comp]
X_test_pca <- predict(pca_model, X_test_deriv)[, 1:n_comp]

reconstructed <- X_train_pca %*% t(pca_model$rotation[, 1:n_comp])
reconstructed <- scale(reconstructed, center = -pca_model$center, scale = 1 / pca_model$scale)
OD <- rowSums((X_train_deriv - reconstructed)^2)
keep_OD <- which(OD <= quantile(OD, 0.975))
X_train_pca <- X_train_pca[keep_OD, ]
y_train <- y_train[keep_OD]

iso <- isolationForest$new(sample_size = min(256, nrow(X_train_pca)), num_trees = 100)
iso$fit(as.data.frame(X_train_pca))
iso_scores <- iso$predict(as.data.frame(X_train_pca))
X_train_pca <- X_train_pca[iso_scores$anomaly_score <= 0.6, ]
y_train <- y_train[iso_scores$anomaly_score <= 0.6]

# ---- 8. Train Base Models ----
cv_ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3, savePredictions = TRUE)
svm_grid <- expand.grid(C = c(1, 10), sigma = c(0.001, 0.01))
models <- list(
    PLSR = train(X_train_pca, y_train, method = "pls", tuneLength = 15, trControl = cv_ctrl),
    Cubist = train(X_train_pca, y_train, method = "cubist", tuneLength = 10, trControl = cv_ctrl),
    SVR = train(X_train_pca, y_train, method = "svmRadial", tuneGrid = svm_grid, trControl = cv_ctrl),
    RF = train(X_train_pca, y_train, method = "ranger", tuneLength = 10, trControl = cv_ctrl),
    PCR = train(X_train_pca, y_train, method = "pcr", tuneLength = 15, trControl = cv_ctrl)
)

# ---- 9. Meta-Learner ----
meta_train <- bind_rows(
    lapply(names(models), function(m) {
        data.frame(ID = models[[m]]$pred$rowIndex, Model = m, Pred = models[[m]]$pred$pred)
    })
) %>% group_by(ID, Model) %>% summarise(Pred = mean(Pred), .groups = "drop") %>%
    pivot_wider(names_from = Model, values_from = Pred) %>% arrange(ID)

meta_train$Target <- y_train[meta_train$ID]
X_meta <- as.matrix(meta_train %>% select(-ID, -Target))
y_meta <- meta_train$Target
meta_ridge <- cv.glmnet(X_meta, y_meta, alpha = 0)

holdout_preds <- bind_rows(
    lapply(names(models), function(m) {
        preds <- if (models[[m]]$method == "ranger") {
            predict(models[[m]]$finalModel, data = X_test_pca)$predictions
        } else {
            predict(models[[m]], newdata = X_test_pca)
        }
        data.frame(Row = seq_along(preds), Model = m, Pred = preds)
    })
) %>% group_by(Row, Model) %>% summarise(Pred = mean(Pred), .groups = "drop") %>%
    pivot_wider(names_from = Model, values_from = Pred) %>% arrange(Row)

X_holdout_meta <- as.matrix(holdout_preds %>% select(-Row))
meta_preds <- predict(meta_ridge, X_holdout_meta, s = "lambda.min")

# ---- 10. Evaluation ----
Observed <- expm1(y_test)
Predicted <- tryCatch(expm1(as.numeric(meta_preds)), error = function(e) rep(NA, length(y_test)))
lambda_min <- meta_ridge$lambda.min

meta_metrics <- data.frame(
    Model = "Meta-Learner",
    R2 = round(cor(Observed, Predicted)^2, 3),
    RMSE = round(rmse(Observed, Predicted), 3),
    MAE = round(mae(Observed, Predicted), 3),
    Bias = round(mean(Predicted - Observed), 3),
    SD = round(sd(Observed), 3),
    RPD = round(sd(Observed) / sqrt(mean((Observed - Predicted)^2)), 3)
)

write.csv(meta_metrics, paste0("outputs/results/meta_learner_metrics_", timestamp, ".csv"), row.names = FALSE)
write.csv(data.frame(Observed = Observed, Predicted = Predicted), paste0("outputs/results/meta_learner_predictions_", timestamp, ".csv"), row.names = FALSE)

# ---- 11. Visualizations ----

meta_long <- meta_metrics %>% pivot_longer(cols = -Model, names_to = "Metric", values_to = "Value")

# Uncertainty Plot
set.seed(123)
n_boot <- 100
boot_preds <- replicate(n_boot, {
    sample_idx <- sample(seq_len(nrow(X_holdout_meta)), replace = TRUE)
    boot_fit <- cv.glmnet(X_meta, y_meta, alpha = 0)
    predict(boot_fit, newx = X_holdout_meta[sample_idx, ], s = "lambda.min")
})

boot_preds <- expm1(boot_preds)
pred_mean <- rowMeans(boot_preds)
pred_sd <- apply(boot_preds, 1, sd)

uncertainty_df <- data.frame(
    Observed = Observed,
    Predicted = Predicted,
    Lower = pmax(pred_mean - 1.96 * pred_sd, 0),
    Upper = pred_mean + 1.96 * pred_sd
)

uncert_plot <- ggplot(uncertainty_df, aes(x = Observed, y = Predicted)) +
    geom_point(color = "#0072B2", alpha = 0.5) +
    geom_errorbar(aes(ymin = Lower, ymax = Upper), color = "gray50", width = 0.2) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
    theme_minimal() +
    labs(title = "Meta-Learner Prediction with 95% Confidence Intervals",
         x = "Observed OM (%)", y = "Predicted OM (%)")

ggsave(paste0("outputs/plots/meta_learner_uncertainty_plot_", timestamp, ".png"), uncert_plot, width = 8, height = 5)

# Scatter Plot with Metrics
scatter_metrics <- paste0(
    "R² = ", round(cor(Observed, Predicted)^2, 3),
    "\nRMSE = ", round(rmse(Observed, Predicted), 3),
    "\nMAE = ", round(mae(Observed, Predicted), 3)
)

scatter_plot <- ggplot(data.frame(Observed, Predicted), aes(x = Observed, y = Predicted)) +
    geom_point(alpha = 0.6, color = "#0072B2") +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
    geom_text(x = min(Observed), y = max(Predicted), label = scatter_metrics,
              hjust = 0, vjust = 1.2, size = 3.5) +
    theme_minimal() +
    labs(title = "Observed vs Predicted (Meta-Learner)",
         x = "Observed OM (%)", y = "Predicted OM (%)")

ggsave(paste0("outputs/plots/meta_learner_observed_vs_predicted_", timestamp, ".png"), scatter_plot, width = 7, height = 5)

# Residuals Plot
residuals_df <- data.frame(Predicted = Predicted, Observed = Observed)
residuals_df$Residual <- residuals_df$Predicted - residuals_df$Observed
residuals_plot <- ggplot(residuals_df, aes(x = Predicted, y = Residual)) +
    geom_point(alpha = 0.5, color = "darkorange") +
    geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
    geom_smooth(method = "loess", se = FALSE, color = "blue") +
    theme_minimal() +
    labs(title = "Residuals vs Predicted Values (Meta-Learner)",
         x = "Predicted OM (%)", y = "Residual")

ggsave(paste0("outputs/plots/meta_learner_residuals_vs_predicted_", timestamp, ".png"), residuals_plot, width = 7, height = 5)

# Bar Plot
bar_plot <- ggplot(meta_long, aes(x = Metric, y = Value, fill = Metric)) +
    geom_bar(stat = "identity", show.legend = FALSE) +
    theme_minimal() +
    labs(title = "Meta-Learner Performance Metrics", x = "Metric", y = "Value")

ggsave(paste0("outputs/plots/meta_learner_barplot_metrics_", timestamp, ".png"), bar_plot, width = 8, height = 5)

# Box Plot
box_plot <- ggplot(meta_long, aes(x = Metric, y = Value, fill = Metric)) +
    geom_boxplot(show.legend = FALSE, outlier.color = "red", alpha = 0.7) +
    theme_minimal() +
    labs(title = "Boxplot of Meta-Learner Metrics", x = "Metric", y = "Value")

ggsave(paste0("outputs/plots/meta_learner_boxplot_metrics_", timestamp, ".png"), box_plot, width = 8, height = 5)

# ---- Done ----
cat("\n✅ Full meta-learner pipeline executed and visualizations saved.\n")




