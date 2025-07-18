# ======================= Soil Spectroscopy Pipeline (Cleaned + Rearranged) =======================
# Load required libraries
library(data.table)
library(dplyr)
library(ggplot2)
library(tidyr)
library(caret)
library(prospectr)
library(baseline)
library(pls)
library(Cubist)
library(kernlab)
library(randomForest)
library(ranger)
library(quantregForest)
library(factoextra)
library(Metrics)
library(doParallel)

# Optional packages
if (!requireNamespace("fastshap", quietly = TRUE)) install.packages("fastshap")
library(fastshap)
if (!requireNamespace("solitude", quietly = TRUE)) install.packages("solitude")
library(solitude)

# Timestamp
timestamp <- format(Sys.time(), "%Y%m%d_%H%M")

# ==================== 1. Load & Filter Data ====================
data <- fread("data/merged_nocf_data.csv")
wl_cols <- grep("^X[0-9]+$", names(data), value = TRUE)
wl <- as.numeric(sub("X", "", wl_cols))

# --- Remove OM outliers using IQR ---
Q1 <- quantile(data$OM, 0.25)
Q3 <- quantile(data$OM, 0.75)
IQR_OM <- Q3 - Q1
lower <- Q1 - 1.5 * IQR_OM
upper <- Q3 + 1.5 * IQR_OM
data <- data[data$OM >= lower & data$OM <= upper, ]

# Transform OM to log1p scale
y <- log1p(data$OM)

# ==================== 2. Spectral Preprocessing ====================

# Trim and resample wavelengths
keep <- wl >= 500 & wl <= 2450
X <- data[, ..wl_cols][, keep, with = FALSE]
wl <- wl[keep]
new_wl <- seq(min(wl), max(wl), by = 5)
X_resampled <- resample(X, wav = wl, new.wav = new_wl)
colnames(X_resampled) <- paste0("W", new_wl)

# Skip baseline correction
X_corrected <- X_resampled

# ==================== 3. Split & Transform ====================
set.seed(42)
split <- createDataPartition(y, p = 0.7, list = FALSE)
X_train_raw <- X_corrected[split, ]
X_test_raw <- X_corrected[-split, ]
y_train <- y[split]
y_test <- y[-split]

# Preprocessing steps
preprocess_chain <- function(mat) {
    mat <- log10(1 / (mat + 1e-8))
    mat <- savitzkyGolay(mat, p = 2, w = 11, m = 0)
    mat <- gapDer(mat, m = 1, w = 11, s = 3)
    mat
}

X_train_deriv <- preprocess_chain(X_train_raw)
X_test_deriv <- preprocess_chain(X_test_raw)

# Improved zero-/low-variance column filtering
non_zero <- apply(X_train_deriv, 2, function(x) is.finite(sd(x)) && sd(x) > 1e-10)
X_train_deriv <- X_train_deriv[, non_zero]
X_test_deriv <- X_test_deriv[, non_zero]
cat("Retained", ncol(X_train_deriv), "spectral features after preprocessing.\n")

# ==================== 4. PCA & Outlier Removal ====================

# Initial PCA
pca_model <- prcomp(X_train_deriv, center = TRUE, scale. = TRUE)
var_exp <- pca_model$sdev^2 / sum(pca_model$sdev^2)

# Scree Plot
scree_data <- data.frame(PC = 1:length(var_exp), Variance = var_exp)
ggplot(scree_data[1:20, ], aes(x = PC, y = Variance)) +
    geom_point() +
    geom_line() +
    ylab("Explained spectral variance") +
    xlab("PC") +
    ggtitle("Scree plot of PCA") +
    theme_minimal()
ggsave(paste0("outputs/plots/pca_scree_plot_", timestamp, ".png"), width = 8, height = 5)

# Cumulative variance to decide component cutoff
cum_exp <- cumsum(var_exp)
n_comp <- which(cum_exp >= 0.99)[1]
X_train_pca <- predict(pca_model, X_train_deriv)[, 1:n_comp]
X_test_pca <- predict(pca_model, X_test_deriv)[, 1:n_comp]

# --- OD Outlier Removal ---
reconstructed <- X_train_pca %*% t(pca_model$rotation[, 1:n_comp])
reconstructed <- scale(reconstructed, center = -pca_model$center, scale = 1 / pca_model$scale)
OD <- rowSums((X_train_deriv - reconstructed)^2)
OD_cut <- quantile(OD, 0.975)
keep_OD <- which(OD <= OD_cut)
X_train_deriv <- X_train_deriv[keep_OD, ]
y_train <- y_train[keep_OD]

# Recompute PCA
pca_model <- prcomp(X_train_deriv, center = TRUE, scale. = TRUE)
var_exp <- pca_model$sdev^2 / sum(pca_model$sdev^2)
cum_exp <- cumsum(var_exp)
n_comp <- which(cum_exp >= 0.99)[1]
X_train_pca <- predict(pca_model, X_train_deriv)[, 1:n_comp]
X_test_pca <- predict(pca_model, X_test_deriv)[, 1:n_comp]

# --- Isolation Forest Outlier Removal ---
iso <- isolationForest$new(sample_size = min(256, nrow(X_train_pca)), num_trees = 100)
iso$fit(as.data.frame(X_train_pca))
iso_scores <- iso$predict(as.data.frame(X_train_pca))
iso_outliers <- which(iso_scores$anomaly_score > 0.6)
X_train_pca <- X_train_pca[-iso_outliers, ]
y_train <- y_train[-iso_outliers]

# Recompute X_test_pca to match final PCA model
X_test_pca <- predict(pca_model, X_test_deriv)[, 1:n_comp]

# Remove rows with NA in X_test_pca
na_rows <- which(apply(X_test_pca, 1, function(x) any(is.na(x))))
if (length(na_rows) > 0) {
    cat("Removing", length(na_rows), "rows with NA from X_test_pca and y_test\n")
    X_test_pca <- X_test_pca[-na_rows, ]
    y_test <- y_test[-na_rows]
}

# ==================== 5. Modeling & Evaluation ====================

# Train QRF
qrf <- quantregForest(X_train_pca, y_train, ntrees = 500)
qrf_ci <- predict(qrf, X_test_pca, what = c(0.05, 0.5, 0.95))

# SHAP values (QRF)
shap_values <- fastshap::explain(qrf, X = X_train_pca, pred_wrapper = function(object, newdata) predict(object, newdata, what = 0.5))
shap_long <- as.data.frame(shap_values) %>%
    mutate(id = row_number()) %>%
    pivot_longer(-id, names_to = "Feature", values_to = "SHAP")

ggplot(shap_long, aes(x = reorder(Feature, abs(SHAP), median), y = SHAP)) +
    geom_boxplot(fill = "skyblue", alpha = 0.7, outlier.alpha = 0.1) +
    coord_flip() +
    labs(title = "SHAP values for QRF (Median prediction)", x = "PCA Features", y = "SHAP value") +
    theme_minimal()
ggsave(paste0("outputs/plots/qrf_shap_boxplot_", timestamp, ".png"), width = 10, height = 6)

# ==================== Continue pipeline... ====================
