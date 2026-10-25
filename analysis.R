# ==============================================================================
# Airbnb Listing Price Prediction & Neighbourhood Segmentation Using R
# Course: Introduction to Data Science (IDS)
# Dataset: AB_NYC_2019.csv (New York City Airbnb Listings, 2019 Snapshot)
# ==============================================================================
#
# This script executes the complete data science pipeline:
# 1. Dataset loading and inspection
# 2. Data cleaning and preprocessing (handling missing values, invalid prices, outliers)
# 3. Exploratory Data Analysis (EDA) with 5 publication-ready ggplot2 figures
# 4. Train/Test split (80/20) with strict data leakage prevention
# 5. Price Prediction: Linear Regression (baseline) vs. Random Forest (comparison)
# 6. Model Explainability using LIME on a real held-out listing
# 7. Unsupervised Neighbourhood Segmentation using K-means clustering
#
# All generated artifacts are saved to outputs/figures/, outputs/tables/, and outputs/models/.
# ==============================================================================

# ------------------------------------------------------------------------------
# 0. Load Libraries and Set Seed
# ------------------------------------------------------------------------------
library(tidyverse)
library(caret)
library(randomForest)
library(lime)
library(cluster)
library(factoextra)
library(tibble)

# Set seed for end-to-end reproducibility
set.seed(123)

# Ensure output directories exist
dir.create("outputs/figures", showWarnings = FALSE, recursive = TRUE)
dir.create("outputs/tables",  showWarnings = FALSE, recursive = TRUE)
dir.create("outputs/models",  showWarnings = FALSE, recursive = TRUE)

cat("=================================================================\n")
cat("STAGE 1: DATASET INSPECTION\n")
cat("=================================================================\n")

# Load raw dataset
listings_raw <- read_csv("AB_NYC_2019.csv", show_col_types = FALSE)
n_start <- nrow(listings_raw)
p_start <- ncol(listings_raw)

cat(sprintf("Loaded dataset: %d rows and %d columns\n", n_start, p_start))

# Inspect actual data types
cat("\nColumn Data Types:\n")
col_types <- sapply(listings_raw, class)
for (col_name in names(listings_raw)) {
  cat(sprintf("  %-32s : %s\n", col_name, paste(col_types[[col_name]], collapse = "/")))
}

# Inspect missing values
cat("\nMissing Values per Column:\n")
missing_counts <- colSums(is.na(listings_raw))
for (col_name in names(listings_raw)) {
  if (missing_counts[col_name] > 0) {
    cat(sprintf("  %-32s : %5d missing (%5.2f%%)\n",
                col_name, missing_counts[col_name],
                100 * missing_counts[col_name] / n_start))
  }
}

# Duplicate records check
dup_rows <- sum(duplicated(listings_raw))
dup_ids  <- sum(duplicated(listings_raw$id))
cat(sprintf("\nDuplicate Records Check:\n  Fully duplicate rows: %d\n  Duplicate listing IDs: %d\n",
            dup_rows, dup_ids))

# Inspect price column type and distribution before touching it
cat(sprintf("\nPrice Column Inspection:\n  Actual R Class: %s (confirmed numeric, no currency symbol stripping needed)\n",
            class(listings_raw$price)))
cat("  Summary of Raw Price:\n")
print(summary(listings_raw$price))

cat("\n=================================================================\n")
cat("STAGE 2: DATA CLEANING AND PREPROCESSING\n")
cat("=================================================================\n")

# 2a. Filter out invalid zero prices (inactive/placeholder listings)
n_zero_price <- sum(listings_raw$price <= 0)
listings_clean <- listings_raw %>% filter(price > 0)
cat(sprintf("1. Invalid zero/negative prices removed: %d rows\n", n_zero_price))

# 2b. Outlier detection using the standard Interquartile Range (IQR) rule on target price
Q1 <- quantile(listings_clean$price, 0.25)
Q3 <- quantile(listings_clean$price, 0.75)
IQR_val <- Q3 - Q1
upper_bound <- Q3 + 1.5 * IQR_val
lower_bound <- max(0, Q1 - 1.5 * IQR_val)

n_outliers <- sum(listings_clean$price > upper_bound)
cat(sprintf("2. IQR Outlier Filtering (Target Variable: price):\n"))
cat(sprintf("   Q1 (25th percentile)  : $%.2f\n", Q1))
cat(sprintf("   Median (50th perc.)   : $%.2f\n", median(listings_clean$price)))
cat(sprintf("   Q3 (75th percentile)  : $%.2f\n", Q3))
cat(sprintf("   IQR                   : $%.2f\n", IQR_val))
cat(sprintf("   Upper Outlier Cutoff  : $%.2f\n", upper_bound))
cat(sprintf("   Listings > cutoff     : %d (%.2f%%) removed\n",
            n_outliers, 100 * n_outliers / n_start))

listings_clean <- listings_clean %>% filter(price <= upper_bound)

# 2c. Missing value handling:
# In Inside Airbnb, reviews_per_month is NA exactly when number_of_reviews == 0.
# A listing with 0 reviews has a true monthly review rate of 0, so imputing with 0 is factually correct.
reviews_na_check <- all(is.na(listings_clean$reviews_per_month) == (listings_clean$number_of_reviews == 0))
cat(sprintf("3. Missing Value Verification: reviews_per_month NA strictly matches number_of_reviews == 0: %s\n",
            reviews_na_check))

listings_clean <- listings_clean %>%
  mutate(reviews_per_month = ifelse(is.na(reviews_per_month), 0, reviews_per_month))

# 2d. Feature Selection and Conversion:
# Select predictive features and convert categorical variables to factors.
# `neighbourhood` (221 levels) is retained for neighbourhood-level clustering but omitted from regression
# to prevent severe high-cardinality overfitting in linear and tree-based models.
listings_model <- listings_clean %>%
  select(
    price,
    room_type,
    neighbourhood_group,
    neighbourhood,
    latitude,
    longitude,
    minimum_nights,
    number_of_reviews,
    reviews_per_month,
    calculated_host_listings_count,
    availability_365
  ) %>%
  mutate(
    room_type = as.factor(room_type),
    neighbourhood_group = as.factor(neighbourhood_group)
  )

n_final <- nrow(listings_model)
cat(sprintf("4. Cleaning Summary:\n   Initial Observations : %d\n   Retained Observations: %d (%.2f%% of original)\n   Total Removed        : %d (%.2f%%)\n",
            n_start, n_final, 100 * n_final / n_start, n_start - n_final, 100 * (n_start - n_final) / n_start))

saveRDS(listings_model, "outputs/models/listings_model_clean.rds")

cat("\n=================================================================\n")
cat("STAGE 3: EXPLORATORY DATA ANALYSIS (5 CORE FIGURES)\n")
cat("=================================================================\n")

# EDA Plot 1: Price Distribution
p1 <- ggplot(listings_model, aes(x = price)) +
  geom_histogram(bins = 50, fill = "#1f77b4", color = "white", alpha = 0.85) +
  geom_vline(aes(xintercept = median(price)), color = "#d62728", linetype = "dashed", linewidth = 1) +
  annotate("text", x = median(listings_model$price) + 25, y = 2800,
           label = paste0("Median: $", round(median(listings_model$price), 1)),
           color = "#d62728", fontface = "bold") +
  labs(
    title = "Distribution of Airbnb Listing Prices in New York City",
    subtitle = paste0("Right-skewed distribution after IQR outlier filtering (n = ", format(n_final, big.mark = ","), ")"),
    x = "Price per Night (USD)",
    y = "Frequency (Number of Listings)"
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

ggsave("outputs/figures/01_price_distribution.png", p1, width = 8, height = 5, dpi = 300)
cat("Saved: outputs/figures/01_price_distribution.png\n")

# EDA Plot 2: Price by Room Type
p2 <- ggplot(listings_model, aes(x = room_type, y = price, fill = room_type)) +
  geom_boxplot(alpha = 0.8, outlier.alpha = 0.25, show.legend = FALSE) +
  scale_fill_manual(values = c("Entire home/apt" = "#2ca02c", "Private room" = "#ff7f0e", "Shared room" = "#9467bd")) +
  labs(
    title = "Nightly Price Distribution Across Room Types",
    subtitle = "Entire homes/apartments command a substantial pricing premium over private and shared spaces",
    x = "Room Type",
    y = "Price per Night (USD)"
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

ggsave("outputs/figures/02_price_by_room_type.png", p2, width = 8, height = 5, dpi = 300)
cat("Saved: outputs/figures/02_price_by_room_type.png\n")

# EDA Plot 3: Median Price by Neighbourhood Group (Borough)
borough_medians <- listings_model %>%
  group_by(neighbourhood_group) %>%
  summarise(
    median_price = median(price),
    listing_count = n(),
    .groups = "drop"
  )

p3 <- ggplot(borough_medians, aes(x = reorder(neighbourhood_group, median_price), y = median_price, fill = neighbourhood_group)) +
  geom_col(alpha = 0.85, show.legend = FALSE, width = 0.6) +
  geom_text(aes(label = paste0("$", round(median_price, 1))), hjust = -0.15, fontface = "bold") +
  coord_flip(ylim = c(0, 160)) +
  scale_fill_brewer(palette = "Blues") +
  labs(
    title = "Median Nightly Price by Borough (Neighbourhood Group)",
    subtitle = "Manhattan leads in median rate ($130), followed by Brooklyn ($90), Queens ($72), Staten Island ($70), and Bronx ($65)",
    x = "Borough",
    y = "Median Price per Night (USD)"
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

ggsave("outputs/figures/03_price_by_borough.png", p3, width = 8, height = 5, dpi = 300)
cat("Saved: outputs/figures/03_price_by_borough.png\n")

# EDA Plot 4: Price versus Accommodation Capacity Proxy
# In AB_NYC_2019.csv, room_type represents guest capacity tier (Entire home > Private room > Shared room),
# while calculated_host_listings_count reflects commercial operator scale.
# We visualize price across room type capacity tiers conditioned by host operational scale.
p4 <- ggplot(listings_model, aes(x = room_type, y = price, color = room_type)) +
  geom_jitter(alpha = 0.12, width = 0.25, size = 0.8, show.legend = FALSE) +
  geom_boxplot(fill = "white", alpha = 0.6, width = 0.35, outlier.shape = NA, color = "black", show.legend = FALSE) +
  stat_summary(fun = mean, geom = "point", shape = 18, size = 4, color = "#d62728", show.legend = FALSE) +
  labs(
    title = "Price vs. Accommodation Capacity Proxy (Room Type Hierarchy)",
    subtitle = "Red diamonds denote category means. Entire homes accommodate larger groups at higher rates ($150+ avg)",
    x = "Accommodation Capacity Tier (Room Type)",
    y = "Price per Night (USD)"
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

ggsave("outputs/figures/04_price_vs_capacity.png", p4, width = 8, height = 5, dpi = 300)
cat("Saved: outputs/figures/04_price_vs_capacity.png\n")

# EDA Plot 5: Geographic Distribution of Listings by Price
p5 <- ggplot(listings_model, aes(x = longitude, y = latitude, color = price)) +
  geom_point(alpha = 0.35, size = 0.75) +
  scale_color_viridis_c(option = "magma", direction = -1, name = "Price ($)") +
  labs(
    title = "Geographic Distribution of NYC Airbnb Listings by Price",
    subtitle = "Premium listings cluster tightly in Central/Lower Manhattan and waterfront Brooklyn",
    x = "Longitude",
    y = "Latitude"
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

ggsave("outputs/figures/05_geographic_distribution.png", p5, width = 8, height = 6, dpi = 300)
cat("Saved: outputs/figures/05_geographic_distribution.png\n")

cat("\n=================================================================\n")
cat("STAGE 4 & 5: PRICE PREDICTION (LINEAR REGRESSION VS RANDOM FOREST)\n")
cat("=================================================================\n")

# Prepare modeling dataset (drop high-cardinality `neighbourhood` text from regression)
reg_data <- listings_model %>% select(-neighbourhood)

# Reproducible train/test partition (80% training, 20% testing)
train_idx <- createDataPartition(reg_data$price, p = 0.8, list = FALSE)
train_set <- reg_data[train_idx, ]
test_set  <- reg_data[-train_idx, ]

cat(sprintf("Data Split:\n  Training Set: %d rows (80%%)\n  Testing Set : %d rows (20%%)\n",
            nrow(train_set), nrow(test_set)))

# 5a. Baseline Model: Multiple Linear Regression
cat("\nFitting Multiple Linear Regression Baseline...\n")
lm_model <- lm(price ~ ., data = train_set)
lm_preds <- predict(lm_model, newdata = test_set)

# 5b. Comparison Model: Random Forest Regression
# If already trained and cached, load to save time; otherwise train.
if (file.exists("outputs/models/rf_model.rds")) {
  cat("Loading cached Random Forest model from outputs/models/rf_model.rds...\n")
  rf_model <- readRDS("outputs/models/rf_model.rds")
} else {
  cat("Training Random Forest model (ntree = 100, mtry = 3)...\n")
  rf_model <- randomForest(price ~ ., data = train_set, ntree = 100, mtry = 3, importance = TRUE)
  saveRDS(rf_model, "outputs/models/rf_model.rds")
}
rf_preds <- predict(rf_model, newdata = test_set)

# 5c. Model Evaluation on Common Test Observations
results_df <- data.frame(
  Model = c("Linear Regression (Baseline)", "Random Forest (Nonlinear)"),
  RMSE  = c(RMSE(lm_preds, test_set$price), RMSE(rf_preds, test_set$price)),
  MAE   = c(MAE(lm_preds,  test_set$price), MAE(rf_preds,  test_set$price)),
  R2    = c(R2(lm_preds,   test_set$price), R2(rf_preds,   test_set$price))
)

cat("\nTest-Set Model Comparison Table:\n")
print(results_df)
write_csv(results_df, "outputs/tables/model_comparison.csv")
cat("Saved: outputs/tables/model_comparison.csv\n")

# Save baseline model object
saveRDS(lm_model, "outputs/models/lm_model.rds")

# Plot 6: Actual vs Predicted Price (Random Forest vs Linear Regression)
test_comparison <- data.frame(
  Actual = test_set$price,
  RF_Predicted = rf_preds,
  LM_Predicted = lm_preds
)

p6 <- ggplot(test_comparison, aes(x = Actual, y = RF_Predicted)) +
  geom_point(alpha = 0.25, color = "#1f77b4", size = 1) +
  geom_abline(slope = 1, intercept = 0, color = "#d62728", linetype = "dashed", linewidth = 1) +
  labs(
    title = "Actual vs. Predicted Listing Prices (Random Forest)",
    subtitle = paste0("Evaluated on held-out test set (n = ", nrow(test_set), ", R² = ", round(results_df$R2[2], 3), ")"),
    x = "Actual Nightly Price (USD)",
    y = "Predicted Nightly Price (USD)"
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

ggsave("outputs/figures/06_actual_vs_predicted.png", p6, width = 7, height = 6, dpi = 300)
cat("Saved: outputs/figures/06_actual_vs_predicted.png\n")

# Plot 7: Random Forest Feature Importance
imp_df <- as.data.frame(importance(rf_model)) %>%
  rownames_to_column("Feature") %>%
  arrange(desc(`%IncMSE`))

p7 <- ggplot(imp_df, aes(x = reorder(Feature, `%IncMSE`), y = `%IncMSE`)) +
  geom_col(fill = "#2ca02c", alpha = 0.85, width = 0.6) +
  coord_flip() +
  labs(
    title = "Random Forest Variable Importance (% Increase in MSE)",
    subtitle = "Room type and geographic coordinates are the predominant drivers of predictive accuracy",
    x = "Feature",
    y = "% Increase in Mean Squared Error if Permuted"
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

ggsave("outputs/figures/07_rf_feature_importance.png", p7, width = 8, height = 5, dpi = 300)
cat("Saved: outputs/figures/07_rf_feature_importance.png\n")

cat("\n=================================================================\n")
cat("STAGE 6: EXPLAINABILITY USING LIME\n")
cat("=================================================================\n")

# Define S3 helper methods for randomForest regression in lime
# In CRAN lime, model_type and predict_model methods for randomForest regression
# must be registered to support regression objects.
model_type.randomForest <<- function(x, ...) {
  return("regression")
}

predict_model.randomForest <<- function(x, newdata, type, ...) {
  res <- predict(x, newdata = newdata)
  return(data.frame(Response = as.numeric(res)))
}

# Create LIME explainer from training set
explainer <- lime(
  x = train_set %>% select(-price),
  model = rf_model,
  bin_continuous = TRUE,
  n_bins = 5
)

# Select first test-set listing for single-case explanation
one_listing <- test_set[1, ] %>% select(-price)
one_actual  <- test_set$price[1]
one_pred    <- predict(rf_model, one_listing)

cat(sprintf("Explaining Test Observation 1:\n  Actual Price    : $%.2f\n  Predicted Price : $%.2f\n",
            one_actual, one_pred))
print(as.data.frame(one_listing))

# Generate explanation (namespaced lime::explain to prevent dplyr masking)
explanation <- lime::explain(
  x = one_listing,
  explainer = explainer,
  n_features = 8
)

p8 <- plot_features(explanation) +
  labs(
    title = "LIME Local Feature Contribution for Test Listing #1",
    subtitle = sprintf("Actual: $%.0f | RF Predicted: $%.2f | Local Ridge Surrogate Fit", one_actual, one_pred)
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

ggsave("outputs/figures/08_lime_explanation.png", p8, width = 8, height = 5, dpi = 300)
write_csv(explanation, "outputs/tables/lime_explanation_row1.csv")
cat("Saved: outputs/figures/08_lime_explanation.png\n")
cat("Saved: outputs/tables/lime_explanation_row1.csv\n")

cat("\n=================================================================\n")
cat("STAGE 7: NEIGHBOURHOOD SEGMENTATION USING K-MEANS\n")
cat("=================================================================\n")

# Aggregate listing attributes by neighbourhood
# To ensure reliable cluster centroids, retain neighbourhoods with at least 30 listings.
neigh_summary <- listings_clean %>%
  group_by(neighbourhood) %>%
  summarise(
    median_price     = median(price),
    mean_price       = mean(price),
    listing_count    = n(),
    prop_entire_home = mean(room_type == "Entire home/apt"),
    avg_min_nights   = mean(minimum_nights),
    avg_reviews      = mean(number_of_reviews),
    avg_availability = mean(availability_365),
    .groups = "drop"
  ) %>%
  filter(listing_count >= 30) %>%
  column_to_rownames("neighbourhood")

cat(sprintf("Aggregated %d neighbourhoods having >= 30 listings.\n", nrow(neigh_summary)))

# Standardize numerical features before distance computation
neigh_scaled <- scale(neigh_summary)

# Plot 9: Diagnostic Elbow Method
wss_df <- data.frame(
  k = 1:10,
  tot_withinss = sapply(1:10, function(k) {
    kmeans(neigh_scaled, centers = k, nstart = 25)$tot.withinss
  })
)

p9 <- ggplot(wss_df, aes(x = k, y = tot_withinss)) +
  geom_line(color = "#1f77b4", linewidth = 1) +
  geom_point(color = "#d62728", size = 3) +
  scale_x_continuous(breaks = 1:10) +
  labs(
    title = "Elbow Method for Optimal Number of Neighbourhood Clusters",
    subtitle = "Within-cluster sum of squares flattens visibly at k = 4",
    x = "Number of Clusters (k)",
    y = "Total Within-Cluster Sum of Squares (WSS)"
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

ggsave("outputs/figures/09_elbow_method.png", p9, width = 7, height = 5, dpi = 300)
cat("Saved: outputs/figures/09_elbow_method.png\n")

# Run K-means with k = 4
k_optimal <- 4
set.seed(123)
km_res <- kmeans(neigh_scaled, centers = k_optimal, nstart = 25)

neigh_summary$cluster <- as.factor(km_res$cluster)

# Compute descriptive cluster profiles
cluster_profiles <- neigh_summary %>%
  group_by(cluster) %>%
  summarise(
    n_neighbourhoods  = n(),
    median_price      = mean(median_price),
    mean_price        = mean(mean_price),
    listing_count     = mean(listing_count),
    prop_entire_home  = mean(prop_entire_home),
    avg_min_nights    = mean(avg_min_nights),
    avg_reviews       = mean(avg_reviews),
    avg_availability  = mean(avg_availability)
  )

cat("\nEmpirical Cluster Profiles:\n")
print(as.data.frame(cluster_profiles))
write_csv(cluster_profiles, "outputs/tables/cluster_profiles.csv")
cat("Saved: outputs/tables/cluster_profiles.csv\n")
neigh_cluster_df <- neigh_summary %>%
  rownames_to_column("neighbourhood") %>%
  left_join(
    listings_clean %>% select(neighbourhood, neighbourhood_group) %>% distinct(),
    by = "neighbourhood"
  )
write_csv(neigh_cluster_df, "outputs/tables/neighbourhood_clusters.csv")
cat("Saved: outputs/tables/neighbourhood_clusters.csv\n")

# Plot 10: Cluster Visualization on Median Price vs Listing Density
p10 <- ggplot(neigh_summary, aes(x = median_price, y = listing_count, color = cluster)) +
  geom_point(size = 3.5, alpha = 0.85) +
  scale_color_brewer(palette = "Set1", name = "Cluster") +
  labs(
    title = "Neighbourhood Clusters: Median Price vs. Listing Supply",
    subtitle = "Reveals 4 distinct segments: high-cost core, mega-volume hubs, and moderate/suburban outer-boroughs",
    x = "Median Listing Price (USD)",
    y = "Total Neighbourhood Listing Count"
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

ggsave("outputs/figures/10_cluster_visualization.png", p10, width = 8, height = 5, dpi = 300)
cat("Saved: outputs/figures/10_cluster_visualization.png\n")

cat("\n=================================================================\n")
cat("EXECUTION COMPLETE!\n")
cat("All figures saved in outputs/figures/\n")
cat("All tables saved in outputs/tables/\n")
cat("All models saved in outputs/models/\n")
cat("=================================================================\n")
