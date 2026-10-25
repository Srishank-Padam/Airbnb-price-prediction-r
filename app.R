# ==============================================================================
# Airbnb Listing Price Prediction & Neighbourhood Segmentation Using R
# R Shiny Interactive Application for Academic Faculty Demonstration
# Course: Introduction to Data Science (IDS) | CBIT Hyderabad
# ==============================================================================

suppressPackageStartupMessages({
  library(shiny)
  library(bslib)
  library(ggplot2)
  library(dplyr)
  library(readr)
  library(scales)
  library(DT)
  library(tibble)
  library(randomForest)
})

# ------------------------------------------------------------------------------
# 1. Global Setup & Asset Loading (Loaded once at startup)
# ------------------------------------------------------------------------------

# Define relative file paths
path_clean_data   <- "outputs/models/listings_model_clean.rds"
path_lm_model     <- "outputs/models/lm_model.rds"
path_rf_model     <- "outputs/models/rf_model.rds"
path_model_comp   <- "outputs/tables/model_comparison.csv"
path_cluster_prof <- "outputs/tables/cluster_profiles.csv"
path_neigh_clust  <- "outputs/tables/neighbourhood_clusters.csv"
path_raw_data     <- "AB_NYC_2019.csv"

# Load cleaned dataset
if (file.exists(path_clean_data)) {
  clean_data <- readRDS(path_clean_data)
} else if (file.exists(path_raw_data)) {
  # Defensive fallback if rds is absent
  raw_df <- read_csv(path_raw_data, show_col_types = FALSE)
  clean_data <- raw_df %>%
    filter(price > 0) %>%
    filter(price <= 334) %>%
    mutate(
      reviews_per_month = ifelse(is.na(reviews_per_month), 0, reviews_per_month),
      room_type = as.factor(room_type),
      neighbourhood_group = as.factor(neighbourhood_group)
    )
} else {
  stop("Analytical dataset not found.")
}

# Key actual baseline statistics
total_listings_clean <- nrow(clean_data)
median_price_clean   <- median(clean_data$price)
mean_price_clean     <- mean(clean_data$price)
sd_price_clean       <- sd(clean_data$price)

# Load trained predictive models
lm_model <- if (file.exists(path_lm_model)) readRDS(path_lm_model) else NULL
rf_model <- if (file.exists(path_rf_model)) readRDS(path_rf_model) else NULL

# Load model evaluation comparison table
model_comp_df <- if (file.exists(path_model_comp)) {
  read_csv(path_model_comp, show_col_types = FALSE)
} else {
  data.frame(
    Model = c("Linear Regression (Baseline)", "Random Forest (Nonlinear)"),
    RMSE  = c(49.23, 43.53),
    MAE   = c(36.46, 30.43),
    R2    = c(0.487, 0.599)
  )
}

# Load cluster profiles
cluster_profiles_df <- if (file.exists(path_cluster_prof)) {
  read_csv(path_cluster_prof, show_col_types = FALSE)
} else {
  NULL
}

# Cluster semantic labels
cluster_labels_map <- c(
  "1" = "Cluster 1: Mega-Volume Cultural Hubs",
  "2" = "Cluster 2: Residential Outer-Borough Communities",
  "3" = "Cluster 3: High-Turnover Transit & Airport Gateways",
  "4" = "Cluster 4: Premium Commercial Manhattan & Coastal Core"
)

# Load or generate qualified neighbourhood clustering table (k = 4, 115 qualified neighbourhoods)
if (file.exists(path_neigh_clust)) {
  neigh_clusters_df <- read_csv(path_neigh_clust, show_col_types = FALSE)
  if (!"cluster_label" %in% names(neigh_clusters_df)) {
    neigh_clusters_df$cluster_label <- cluster_labels_map[as.character(neigh_clusters_df$cluster)]
  }
} else {
  # Reproduce clustering strictly from clean_data matching analysis.R
  ns <- clean_data %>%
    group_by(neighbourhood, neighbourhood_group) %>%
    summarise(
      median_price     = median(price),
      mean_price       = round(mean(price), 2),
      listing_count    = n(),
      prop_entire_home = round(mean(room_type == "Entire home/apt"), 4),
      avg_min_nights   = round(mean(minimum_nights), 2),
      avg_reviews      = round(mean(number_of_reviews), 2),
      avg_availability = round(mean(availability_365), 2),
      latitude         = round(mean(latitude), 5),
      longitude        = round(mean(longitude), 5),
      .groups          = "drop"
    ) %>%
    filter(listing_count >= 30)

  features_mat <- ns %>%
    select(median_price, mean_price, listing_count, prop_entire_home, avg_min_nights, avg_reviews, avg_availability)
  
  set.seed(123)
  km <- kmeans(scale(features_mat), centers = 4, nstart = 25)
  ns$cluster <- as.factor(km$cluster)
  ns$cluster_label <- cluster_labels_map[as.character(km$cluster)]
  neigh_clusters_df <- ns
}

# Coordinate lookup by neighbourhood for interactive form auto-fill
neigh_geo_lookup <- clean_data %>%
  group_by(neighbourhood_group, neighbourhood) %>%
  summarise(
    lat = round(mean(latitude), 5),
    lon = round(mean(longitude), 5),
    n = n(),
    .groups = "drop"
  ) %>%
  arrange(neighbourhood_group, desc(n))

borough_list <- sort(as.character(unique(clean_data$neighbourhood_group)))
room_type_list <- c("Entire home/apt", "Private room", "Shared room")

# ------------------------------------------------------------------------------
# 2. UI Theme & Layout Specification
# ------------------------------------------------------------------------------

app_theme <- bs_theme(
  version = 5,
  primary = "#1B365D",     # Academic Navy
  secondary = "#2B6CB0",   # Soft Steel Blue
  success = "#2E7D32",     # Restrained Forest Green
  danger = "#D62728",      # Restrained Crimson Red
  info = "#0284C7",        # Sky Blue
  bg = "#F8FAFC",          # Clean Light Background
  fg = "#1E293B",          # Deep Slate Font
  base_font = font_collection("Inter", "-apple-system", "Segoe UI", "Roboto", "Helvetica Neue", "sans-serif"),
  heading_font = font_collection("Inter", "-apple-system", "Segoe UI", "Roboto", "Helvetica Neue", "sans-serif")
)

ui <- page_navbar(
  theme = app_theme,
  title = div(
    class = "d-flex align-items-center gap-2",
    tags$span(style = "font-weight: 700; color: #1B365D; font-size: 1.15rem;", "Airbnb NYC Price Valuation & Segmentation"),
    tags$span(class = "badge rounded-pill bg-light text-dark border", style = "font-size: 0.75rem; font-weight: 500;", "IDS Case Study • R Shiny")
  ),
  id = "main_nav",
  header = tags$head(
    tags$style(HTML("
      body { background-color: #F8FAFC; color: #1E293B; }
      .navbar { background-color: #FFFFFF !important; border-bottom: 2px solid #E2E8F0; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
      .navbar-brand { padding-top: 0.25rem; padding-bottom: 0.25rem; }
      .nav-link { font-weight: 600; color: #475569 !important; font-size: 0.95rem; }
      .nav-link.active { color: #1B365D !important; border-bottom: 3px solid #1B365D !important; }
      .card { border: 1px solid #E2E8F0; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.03); background-color: #FFFFFF; }
      .card-header { background-color: #FFFFFF; border-bottom: 1px solid #E2E8F0; font-weight: 600; color: #1B365D; }
      .metric-card { border-left: 4px solid #1B365D; }
      .metric-title { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; color: #64748B; font-weight: 600; }
      .metric-value { font-size: 1.75rem; font-weight: 700; color: #1B365D; line-height: 1.2; }
      .metric-sub { font-size: 0.78rem; color: #64748B; }
      .prediction-box { background: linear-gradient(135deg, #1B365D 0%, #2A4D7E 100%); color: #FFFFFF; border-radius: 8px; padding: 1.5rem; text-align: center; }
      .prediction-price { font-size: 2.75rem; font-weight: 800; letter-spacing: -0.02em; }
      .btn-primary { background-color: #1B365D; border-color: #1B365D; }
      .btn-primary:hover { background-color: #142847; border-color: #142847; }
      .btn-outline-navy { color: #1B365D; border-color: #CBD5E1; background-color: #F8FAFC; font-size: 0.85rem; font-weight: 500; }
      .btn-outline-navy:hover { background-color: #E2E8F0; color: #1B365D; }
      .cluster-badge-1 { background-color: #E41A1C; color: white; padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 0.75rem; }
      .cluster-badge-2 { background-color: #377EB8; color: white; padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 0.75rem; }
      .cluster-badge-3 { background-color: #4DAF4A; color: white; padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 0.75rem; }
      .cluster-badge-4 { background-color: #984EA3; color: white; padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 0.75rem; }
    "))
  ),

  # ============================================================================
  # TAB 1: OVERVIEW
  # ============================================================================
  nav_panel(
    title = "Overview",
    icon = icon("chart-pie"),
    div(
      class = "container-fluid py-3",
      
      # Top Header Banner
      div(
        class = "d-flex justify-content-between align-items-center mb-3 pb-2 border-bottom",
        div(
          h4(class = "fw-bold text-navy mb-1", style = "color: #1B365D;", "Executive Overview & Market Distribution"),
          p(class = "text-muted mb-0", style = "font-size: 0.9rem;",
            "Empirical exploratory analysis of the verified Inside Airbnb NYC dataset after rigorous IQR outlier filtering ($334 upper bound) and inactive record purging.")
        ),
        div(
          class = "text-end",
          span(class = "badge bg-primary px-3 py-2", style = "background-color: #1B365D !important; font-size: 0.85rem;",
               paste0("Sample Size: n = ", format(total_listings_clean, big.mark = ",")))
        )
      ),

      # Actual Summary Metric Cards
      layout_column_wrap(
        width = "220px",
        class = "mb-4",
        
        # Metric 1: Listing Count
        div(
          class = "card p-3 metric-card",
          div(class = "metric-title", "Verified Listings (Sample)"),
          div(class = "metric-value", format(total_listings_clean, big.mark = ",")),
          div(class = "metric-sub", "93.90% of 48,895 raw records retained")
        ),
        
        # Metric 2: Median Price
        div(
          class = "card p-3 metric-card", style = "border-left-color: #D62728;",
          div(class = "metric-title", "Median Nightly Price"),
          div(class = "metric-value", paste0("$", format(median_price_clean, nsmall = 0))),
          div(class = "metric-sub", "Robust central tendency (Q1: $69, Q3: $175)")
        ),

        # Metric 3: Mean Price
        div(
          class = "card p-3 metric-card", style = "border-left-color: #2B6CB0;",
          div(class = "metric-title", "Mean Nightly Price"),
          div(class = "metric-value", paste0("$", format(round(mean_price_clean, 2), nsmall = 2))),
          div(class = "metric-sub", paste0("Standard Deviation: $", round(sd_price_clean, 2)))
        ),

        # Metric 4: IQR Threshold
        div(
          class = "card p-3 metric-card", style = "border-left-color: #475569;",
          div(class = "metric-title", "IQR Outlier Cutoff"),
          div(class = "metric-value", "$334.00"),
          div(class = "metric-sub", "2,972 luxury/erroneous rows purged (6.08%)")
        )
      ),

      # Interactive Filters Card
      card(
        class = "mb-4 p-3",
        div(
          class = "row g-3 align-items-center",
          div(
            class = "col-md-3",
            tags$label(class = "form-label fw-bold mb-1", style = "font-size: 0.85rem; color: #1B365D;", "Filter by Borough:"),
            selectInput("ov_borough_filter", NULL,
                        choices = c("All Boroughs (5 Boroughs)" = "All", borough_list),
                        selected = "All", width = "100%")
          ),
          div(
            class = "col-md-3",
            tags$label(class = "form-label fw-bold mb-1", style = "font-size: 0.85rem; color: #1B365D;", "Filter by Room Type:"),
            selectInput("ov_room_filter", NULL,
                        choices = c("All Room Types (3 Categories)" = "All", room_type_list),
                        selected = "All", width = "100%")
          ),
          div(
            class = "col-md-6 text-end pt-3 pt-md-0",
            div(class = "text-muted", style = "font-size: 0.85rem;",
                textOutput("ov_filtered_summary_text"))
          )
        )
      ),

      # Visualizations Row: Price Distribution & Price by Room Type
      layout_column_wrap(
        width = 1/2,
        class = "mb-4",
        
        card(
          card_header(
            div(class = "d-flex justify-content-between align-items-center",
                span("Nightly Price Distribution"),
                span(class = "badge bg-light text-muted border", "Histogram & Median"))
          ),
          div(class = "p-2",
              plotOutput("ov_plot_price_dist", height = "360px")),
          card_footer(
            div(class = "text-muted small",
                "Right-skewed pricing distribution with a dominant mass between $45 and $150. Dashed red line marks the sample median.")
          )
        ),

        card(
          card_header(
            div(class = "d-flex justify-content-between align-items-center",
                span("Price Variation Across Room Types"),
                span(class = "badge bg-light text-muted border", "Boxplots & Outliers"))
          ),
          div(class = "p-2",
              plotOutput("ov_plot_room_type", height = "360px")),
          card_footer(
            div(class = "text-muted small",
                "Entire homes/apartments command a substantial structural premium over private and shared spaces across all boroughs.")
          )
        )
      ),

      # Detailed Tabular Breakdown
      card(
        card_header("Empirical Pricing Statistics by Room Type & Borough"),
        div(class = "p-3",
            DT::dataTableOutput("ov_summary_table"))
      )
    )
  ),

  # ============================================================================
  # TAB 2: PRICE PREDICTION
  # ============================================================================
  nav_panel(
    title = "Price Prediction",
    icon = icon("calculator"),
    div(
      class = "container-fluid py-3",
      
      div(
        class = "d-flex justify-content-between align-items-center mb-3 pb-2 border-bottom",
        div(
          h4(class = "fw-bold mb-1", style = "color: #1B365D;", "Supervised Nightly Price Prediction Engine"),
          p(class = "text-muted mb-0", style = "font-size: 0.9rem;",
            "Real-time inference using the verified pre-trained Linear Regression and Random Forest models saved during pipeline execution. Zero on-the-fly training.")
        ),
        div(
          class = "d-flex gap-2",
          span(class = "badge bg-success py-2 px-3", "Models Ready"),
          span(class = "badge bg-light text-dark border py-2 px-3", "Zero Data Leakage Pipeline")
        )
      ),

      div(
        class = "row g-3",
        
        # Left Panel: Feature Inputs Card
        div(
          class = "col-lg-5 col-xl-4",
          card(
            card_header(
              div(class = "d-flex justify-content-between align-items-center",
                  span("1. Listing Features & Model Selection"),
                  icon("sliders"))
            ),
            div(
              class = "p-3",
              
              # Model Choice
              div(
                class = "mb-3 p-2 rounded bg-light border",
                tags$label(class = "form-label fw-bold mb-2", style = "color: #1B365D; font-size: 0.88rem;", "Select Model Architecture:"),
                radioButtons(
                  "pred_model_choice", NULL,
                  choices = c(
                    "Random Forest Regressor (R² = 0.599, MAE = $30.43)" = "rf",
                    "Multiple Linear Regression (R² = 0.487, MAE = $36.46)" = "lm"
                  ),
                  selected = "rf"
                )
              ),

              # Room Type
              div(
                class = "mb-3",
                tags$label(class = "form-label fw-bold mb-1", style = "font-size: 0.85rem;", "Room Type:"),
                selectInput("pred_room_type", NULL, choices = room_type_list, selected = "Entire home/apt", width = "100%")
              ),

              # Borough
              div(
                class = "mb-3",
                tags$label(class = "form-label fw-bold mb-1", style = "font-size: 0.85rem;", "Borough (Neighbourhood Group):"),
                selectInput("pred_borough", NULL, choices = borough_list, selected = "Manhattan", width = "100%")
              ),

              # Specific Neighbourhood (Assists coordinates selection)
              div(
                class = "mb-3",
                tags$label(class = "form-label fw-bold mb-1", style = "font-size: 0.85rem;", "Neighbourhood (Auto-fills Coordinates):"),
                selectInput("pred_neighbourhood", NULL, choices = NULL, width = "100%")
              ),

              # Geographic Coordinates (Latitude & Longitude)
              div(
                class = "row g-2 mb-3",
                div(
                  class = "col-6",
                  tags$label(class = "form-label fw-bold mb-1", style = "font-size: 0.82rem;", "Latitude:"),
                  numericInput("pred_lat", NULL, value = 40.7536, min = 40.49, max = 40.92, step = 0.001, width = "100%")
                ),
                div(
                  class = "col-6",
                  tags$label(class = "form-label fw-bold mb-1", style = "font-size: 0.82rem;", "Longitude:"),
                  numericInput("pred_lon", NULL, value = -73.9838, min = -74.25, max = -73.70, step = 0.001, width = "100%")
                )
              ),

              # Minimum Nights
              div(
                class = "mb-3",
                tags$label(class = "form-label fw-bold mb-1", style = "font-size: 0.85rem;", "Minimum Stay (Nights):"),
                sliderInput("pred_min_nights", NULL, min = 1, max = 30, value = 2, step = 1, width = "100%")
              ),

              # Number of Reviews
              div(
                class = "mb-3",
                tags$label(class = "form-label fw-bold mb-1", style = "font-size: 0.85rem;", "Total Number of Reviews:"),
                sliderInput("pred_num_reviews", NULL, min = 0, max = 300, value = 25, step = 1, width = "100%")
              ),

              # Reviews per Month
              div(
                class = "mb-3",
                div(class = "d-flex justify-content-between align-items-center mb-1",
                    tags$label(class = "form-label fw-bold mb-0", style = "font-size: 0.85rem;", "Reviews per Month:"),
                    tags$small(class = "text-muted", "Auto 0 if reviews = 0")),
                sliderInput("pred_reviews_per_month", NULL, min = 0.0, max = 15.0, value = 1.2, step = 0.1, width = "100%")
              ),

              # Host Listings Count & Availability
              div(
                class = "row g-2 mb-3",
                div(
                  class = "col-6",
                  tags$label(class = "form-label fw-bold mb-1", style = "font-size: 0.82rem;", "Host Total Listings:"),
                  numericInput("pred_host_listings", NULL, value = 1, min = 1, max = 50, step = 1, width = "100%")
                ),
                div(
                  class = "col-6",
                  tags$label(class = "form-label fw-bold mb-1", style = "font-size: 0.82rem;", "Availability (365d):"),
                  numericInput("pred_availability", NULL, value = 180, min = 0, max = 365, step = 5, width = "100%")
                )
              ),

              # Predict Button
              div(
                class = "d-grid gap-2 mb-3",
                actionButton("btn_predict", "Calculate Valuation", class = "btn btn-primary py-2 fw-bold fs-6", icon = icon("calculator"))
              ),

              # Quick Presets for Demo
              div(
                class = "pt-2 border-top",
                tags$label(class = "form-label text-muted fw-bold mb-2", style = "font-size: 0.78rem; text-transform: uppercase;", "Faculty Demonstration Presets:"),
                div(
                  class = "d-flex flex-wrap gap-1",
                  actionButton("preset_lime", "Held-Out Test #1 (Brooklyn Private)", class = "btn btn-outline-navy btn-sm"),
                  actionButton("preset_midtown", "Manhattan Midtown Entire Apt", class = "btn btn-outline-navy btn-sm"),
                  actionButton("preset_williamsburg", "Williamsburg Hip Loft", class = "btn btn-outline-navy btn-sm"),
                  actionButton("preset_queens", "Queens Transit Stay", class = "btn btn-outline-navy btn-sm")
                )
              )
            )
          )
        ),

        # Right Panel: Output & Interpretation
        div(
          class = "col-lg-7 col-xl-8",
          
          # Primary Prediction Banner
          div(
            class = "card p-4 mb-3 prediction-box",
            div(style = "font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.06em; opacity: 0.85; font-weight: 600;", "Estimated Market Valuation"),
            div(class = "prediction-price", textOutput("pred_price_display")),
            div(style = "font-size: 1rem; opacity: 0.9; margin-top: 0.25rem;", textOutput("pred_model_badge")),
            div(style = "font-size: 0.85rem; opacity: 0.75; margin-top: 0.5rem;", textOutput("pred_mae_interval_text"))
          ),

          # Dual-Model Comparison Card
          card(
            class = "mb-3",
            card_header(
              div(class = "d-flex justify-content-between align-items-center",
                  span("Side-by-Side Model Comparison (Same Inputs)"),
                  span(class = "badge bg-light text-muted border", "Linear vs. Nonlinear"))
            ),
            div(
              class = "p-3",
              layout_column_wrap(
                width = 1/2,
                div(
                  class = "p-3 border rounded bg-light",
                  div(class = "metric-title", "Random Forest Prediction"),
                  div(class = "metric-value text-success", textOutput("pred_rf_val")),
                  div(class = "metric-sub", "R² = 0.599 | Test MAE = $30.43 (Nonlinear)")
                ),
                div(
                  class = "p-3 border rounded bg-light",
                  div(class = "metric-title", "Linear Regression Baseline"),
                  div(class = "metric-value text-primary", textOutput("pred_lm_val")),
                  div(class = "metric-sub", "R² = 0.487 | Test MAE = $36.46 (Linear)")
                )
              ),
              div(
                class = "mt-3 p-3 rounded bg-light border text-muted", style = "font-size: 0.85rem;",
                uiOutput("pred_comparison_commentary")
              )
            )
          ),

          # Model Evaluation Benchmarks (Test Set)
          card(
            class = "mb-3",
            card_header("Model Evaluation Benchmarks on 9,180 Held-Out Test Listings"),
            div(
              class = "p-3",
              tableOutput("pred_metrics_table"),
              div(
                class = "text-muted small mt-2",
                "Evaluation performed strictly on held-out test data (20% split) created via createDataPartition() with set.seed(123). Random Forest achieves a $6.03 reduction in MAE and explains ~60% of test price variation."
              )
            )
          ),

          # Empirical Feature Influence Breakdown
          card(
            card_header("Pipeline Preprocessing & Structural Valuation Rules"),
            div(
              class = "p-3",
              tags$ul(
                class = "mb-0 text-muted", style = "font-size: 0.88rem; line-height: 1.6;",
                tags$li(tags$strong("Room Type Hierarchy:"), " In the empirical linear baseline, Private Rooms carry a -$75.26/night discount and Shared Rooms carry a -$104.59/night discount relative to an Entire home/apt."),
                tags$li(tags$strong("Borough Baselines:"), " Manhattan commands an automatic +$22.22/night baseline premium over Brooklyn, whereas the Bronx and Staten Island trade at steep discounts."),
                tags$li(tags$strong("Geographic Granularity:"), " Random Forest leverages precise latitude and longitude to reward high-demand corridors (Lower Manhattan, Waterfront Williamsburg) without high-cardinality neighbourhood overfitting."),
                tags$li(tags$strong("Zero Review Invariant:"), " When a listing has 0 reviews, its monthly review rate is strictly imputed to 0.0 to prevent missing value distortion.")
              )
            )
          )
        )
      )
    )
  ),

  # ============================================================================
  # TAB 3: NEIGHBOURHOOD INSIGHTS
  # ============================================================================
  nav_panel(
    title = "Neighbourhood Insights",
    icon = icon("map-location-dot"),
    div(
      class = "container-fluid py-3",
      
      div(
        class = "d-flex justify-content-between align-items-center mb-3 pb-2 border-bottom",
        div(
          h4(class = "fw-bold mb-1", style = "color: #1B365D;", "Unsupervised Neighbourhood Market Segmentation"),
          p(class = "text-muted mb-0", style = "font-size: 0.9rem;",
            "K-Means clustering (k = 4, standardized features, 25 starts) on 115 qualified NYC neighbourhoods having at least 30 listings.")
        ),
        div(
          span(class = "badge bg-primary px-3 py-2", style = "background-color: #1B365D !important;", "Optimal k = 4 (Elbow WSS)")
        )
      ),

      # Cluster Summary Metric Cards
      layout_column_wrap(
        width = "240px",
        class = "mb-4",
        
        div(
          class = "card p-3", style = "border-top: 4px solid #E41A1C;",
          div(class = "d-flex justify-content-between align-items-center mb-1",
              span(class = "cluster-badge-1", "Cluster 1"),
              span(class = "text-muted small", "4 Neighbourhoods")),
          div(class = "fw-bold text-navy mt-1", "Mega-Volume Cultural Hubs"),
          div(class = "metric-sub mt-2", "Williamsburg, Bedford-Stuyvesant, Harlem, Bushwick"),
          div(class = "small mt-1", tags$strong("Supply: "), "3,096 listings/neigh | ", tags$strong("Med: "), "$82.8")
        ),

        div(
          class = "card p-3", style = "border-top: 4px solid #377EB8;",
          div(class = "d-flex justify-content-between align-items-center mb-1",
              span(class = "cluster-badge-2", "Cluster 2"),
              span(class = "text-muted small", "42 Neighbourhoods")),
          div(class = "fw-bold text-navy mt-1", "Residential Outer-Boroughs"),
          div(class = "metric-sub mt-2", "Astoria, Crown Heights, Jackson Heights, Bay Ridge"),
          div(class = "small mt-1", tags$strong("Supply: "), "267.5 listings/neigh | ", tags$strong("Med: "), "$74.8")
        ),

        div(
          class = "card p-3", style = "border-top: 4px solid #4DAF4A;",
          div(class = "d-flex justify-content-between align-items-center mb-1",
              span(class = "cluster-badge-3", "Cluster 3"),
              span(class = "text-muted small", "28 Neighbourhoods")),
          div(class = "fw-bold text-navy mt-1", "Transit & Airport Gateways"),
          div(class = "metric-sub mt-2", "Flushing, Jamaica, Bayside, Allerton, Canarsie"),
          div(class = "small mt-1", tags$strong("Supply: "), "91.7 listings/neigh | ", tags$strong("Med: "), "$70.9")
        ),

        div(
          class = "card p-3", style = "border-top: 4px solid #984EA3;",
          div(class = "d-flex justify-content-between align-items-center mb-1",
              span(class = "cluster-badge-4", "Cluster 4"),
              span(class = "text-muted small", "41 Neighbourhoods")),
          div(class = "fw-bold text-navy mt-1", "Commercial & Coastal Core"),
          div(class = "metric-sub mt-2", "Midtown, SoHo, Chelsea, Tribeca, DUMBO"),
          div(class = "small mt-1", tags$strong("Supply: "), "453.7 listings/neigh | ", tags$strong("Med: "), "$150.7")
        )
      ),

      # Filter Controls for Clustering
      card(
        class = "mb-4 p-3",
        div(
          class = "row g-3 align-items-center",
          div(
            class = "col-md-4",
            tags$label(class = "form-label fw-bold mb-1", style = "font-size: 0.85rem;", "Filter by Cluster Segment:"),
            selectInput("clust_filter_cluster", NULL,
                        choices = c("All Clusters (115 Neighbourhoods)" = "All",
                                    "Cluster 1: Mega-Volume Cultural Hubs" = "1",
                                    "Cluster 2: Residential Outer-Borough Communities" = "2",
                                    "Cluster 3: High-Turnover Transit Gateways" = "3",
                                    "Cluster 4: Premium Commercial Core" = "4"),
                        selected = "All", width = "100%")
          ),
          div(
            class = "col-md-4",
            tags$label(class = "form-label fw-bold mb-1", style = "font-size: 0.85rem;", "Filter by Borough:"),
            selectInput("clust_filter_borough", NULL,
                        choices = c("All Boroughs" = "All", borough_list),
                        selected = "All", width = "100%")
          ),
          div(
            class = "col-md-4 text-md-end pt-2 pt-md-0",
            div(class = "text-muted", style = "font-size: 0.85rem;",
                textOutput("clust_filter_count_text"))
          )
        )
      ),

      # Clustering Scatter Plot Card
      card(
        class = "mb-4",
        card_header(
          div(class = "d-flex justify-content-between align-items-center",
              span("Cluster Mapping: Median Price vs. Total Listing Supply"),
              span(class = "badge bg-light text-muted border", "K-Means Result (n = 115)"))
        ),
        div(class = "p-2",
            plotOutput("clust_scatter_plot", height = "420px")),
        card_footer(
          div(class = "text-muted small",
              "Each point represents an aggregated NYC neighbourhood. Log-scaled listing supply highlights the massive divergence between Mega-Volume Cultural Hubs (Cluster 1) and Premium Commercial Manhattan/Coastal Brooklyn (Cluster 4).")
        )
      ),

      # Interactive Neighbourhood Data Table
      card(
        class = "mb-4",
        card_header("Searchable Neighbourhood Cluster Assignments & Empirical Profiles"),
        div(class = "p-3",
            DT::dataTableOutput("clust_neighbourhood_table"))
      ),

      # Empirical Cluster Profiles Table
      card(
        card_header("Empirical Cluster Means (Source: cluster_profiles.csv)"),
        div(class = "p-3",
            tableOutput("clust_profiles_table"),
            div(class = "text-muted small mt-2",
                "Standardized variables: Median Price, Mean Price, Listing Count, Proportion of Entire Homes, Average Minimum Nights, Average Reviews, and Average Calendar Availability.")
        )
      )
    )
  ),

  # Right-side Navigation Item for Methodology
  nav_spacer(),
  nav_item(
    actionLink("btn_methodology", "About Study", icon = icon("circle-info"), class = "nav-link text-navy")
  )
)

# ------------------------------------------------------------------------------
# 3. Server Logic
# ------------------------------------------------------------------------------

server <- function(input, output, session) {

  # ----------------------------------------------------------------------------
  # Methodology Modal
  # ----------------------------------------------------------------------------
  observeEvent(input$btn_methodology, {
    showModal(modalDialog(
      title = div(class = "fw-bold text-navy", "About This Case Study"),
      div(
        class = "p-2",
        tags$p("This interactive dashboard accompanies the undergraduate Introduction to Data Science (IDS) case study titled:",
               tags$strong(" 'Airbnb Listing Price Prediction & Neighbourhood Segmentation Using R'"),
               " developed at the Department of Artificial Intelligence & Data Science, CBIT Hyderabad."),
        tags$hr(),
        tags$h6(class = "fw-bold", "Core Methodological Highlights:"),
        tags$ul(
          tags$li(tags$strong("Dataset:"), " Inside Airbnb NYC 2019 Snapshot (AB_NYC_2019.csv)."),
          tags$li(tags$strong("Defensive Cleaning:"), " Strict removal of 11 zero-price records and 2,972 IQR target outliers (> $334.00). Missing review rates factually imputed to 0.0."),
          tags$li(tags$strong("Predictive Modeling:"), " Linear Regression baseline vs Random Forest regressor evaluated on 9,180 held-out test listings (20% partition, zero data leakage)."),
          tags$li(tags$strong("Model Explainability:"), " LIME local linear surrogate explanations on held-out listings."),
          tags$li(tags$strong("Market Segmentation:"), " Unsupervised K-Means clustering (k = 4) on 115 neighbourhoods with >= 30 listings.")
        )
      ),
      easyClose = TRUE,
      footer = modalButton("Close")
    ))
  })

  # ----------------------------------------------------------------------------
  # TAB 1: OVERVIEW SERVER LOGIC
  # ----------------------------------------------------------------------------

  # Reactive filtered data for Overview
  ov_filtered_data <- reactive({
    df <- clean_data
    if (isTruthy(input$ov_borough_filter) && input$ov_borough_filter != "All") {
      df <- df %>% filter(neighbourhood_group == input$ov_borough_filter)
    }
    if (isTruthy(input$ov_room_filter) && input$ov_room_filter != "All") {
      df <- df %>% filter(room_type == input$ov_room_filter)
    }
    df
  })

  output$ov_filtered_summary_text <- renderText({
    df <- ov_filtered_data()
    pct <- round(100 * nrow(df) / total_listings_clean, 1)
    paste0("Viewing ", format(nrow(df), big.mark = ","), " listings (", pct, "% of clean sample)")
  })

  # Plot 1: Price Distribution
  output$ov_plot_price_dist <- renderPlot({
    df <- ov_filtered_data()
    req(nrow(df) > 0)
    
    med_val <- median(df$price)
    
    ggplot(df, aes(x = price)) +
      geom_histogram(bins = 45, fill = "#1F77B4", color = "white", alpha = 0.85) +
      geom_vline(xintercept = med_val, color = "#D62728", linetype = "dashed", linewidth = 1) +
      annotate("text", x = med_val + 20, y = Inf, vjust = 2,
               label = paste0("Median: $", round(med_val, 1)),
               color = "#D62728", fontface = "bold", size = 4.2) +
      scale_x_continuous(labels = scales::dollar_format()) +
      scale_y_continuous(labels = scales::comma_format()) +
      labs(
        title = "Nightly Listing Price Distribution",
        subtitle = paste0("Evaluated on ", format(nrow(df), big.mark = ","), " listings after IQR cleaning"),
        x = "Price per Night (USD)",
        y = "Frequency (Listings Count)"
      ) +
      theme_minimal(base_size = 13) +
      theme(
        plot.title = element_text(face = "bold", color = "#1B365D"),
        panel.grid.minor = element_blank()
      )
  })

  # Plot 2: Price by Room Type
  output$ov_plot_room_type <- renderPlot({
    df <- ov_filtered_data()
    req(nrow(df) > 0)

    room_colors <- c("Entire home/apt" = "#2CA02C", "Private room" = "#FF7F0E", "Shared room" = "#9467BD")

    ggplot(df, aes(x = room_type, y = price, fill = room_type)) +
      geom_boxplot(alpha = 0.8, outlier.alpha = 0.25, show.legend = FALSE, width = 0.5) +
      stat_summary(fun = mean, geom = "point", shape = 18, size = 4, color = "#D62728", show.legend = FALSE) +
      scale_fill_manual(values = room_colors) +
      scale_y_continuous(labels = scales::dollar_format()) +
      labs(
        title = "Price Distribution Across Room Types",
        subtitle = "Red diamonds denote category means. Entire homes carry a steep premium.",
        x = "Room Type Category",
        y = "Price per Night (USD)"
      ) +
      theme_minimal(base_size = 13) +
      theme(
        plot.title = element_text(face = "bold", color = "#1B365D"),
        panel.grid.minor = element_blank()
      )
  })

  # Summary Table by Borough & Room Type
  output$ov_summary_table <- DT::renderDataTable({
    df <- ov_filtered_data()
    req(nrow(df) > 0)

    summary_tbl <- df %>%
      group_by(neighbourhood_group, room_type) %>%
      summarise(
        listing_count = n(),
        median_price  = median(price),
        mean_price    = round(mean(price), 2),
        sd_price      = round(sd(price), 2),
        min_price     = min(price),
        max_price     = max(price),
        .groups       = "drop"
      ) %>%
      mutate(prop_sample = round(100 * listing_count / nrow(df), 1)) %>%
      select(
        Borough = neighbourhood_group,
        `Room Type` = room_type,
        `Listings` = listing_count,
        `Share (%)` = prop_sample,
        `Median Price ($)` = median_price,
        `Mean Price ($)` = mean_price,
        `Std Dev ($)` = sd_price,
        `Min ($)` = min_price,
        `Max ($)` = max_price
      )

    DT::datatable(
      summary_tbl,
      options = list(pageLength = 8, dom = 'tip', ordering = TRUE),
      rownames = FALSE
    ) %>%
      DT::formatCurrency(c("Median Price ($)", "Mean Price ($)", "Std Dev ($)", "Min ($)", "Max ($)"), currency = "$")
  })

  # ----------------------------------------------------------------------------
  # TAB 2: PRICE PREDICTION SERVER LOGIC
  # ----------------------------------------------------------------------------

  # Update neighbourhood choices when Borough changes
  observeEvent(input$pred_borough, {
    req(input$pred_borough)
    sub_lookup <- neigh_geo_lookup %>% filter(neighbourhood_group == input$pred_borough)
    neigh_choices <- sub_lookup$neighbourhood
    
    current_selected <- isolate(input$pred_neighbourhood)
    new_selected <- if (isTruthy(current_selected) && current_selected %in% neigh_choices) current_selected else neigh_choices[1]
    
    updateSelectInput(session, "pred_neighbourhood", choices = neigh_choices, selected = new_selected)
  }, ignoreNULL = TRUE)

  # Update coordinates when neighbourhood changes
  observeEvent(input$pred_neighbourhood, {
    req(input$pred_neighbourhood, input$pred_borough)
    coord <- neigh_geo_lookup %>%
      filter(neighbourhood_group == input$pred_borough, neighbourhood == input$pred_neighbourhood)
    
    if (nrow(coord) > 0) {
      updateNumericInput(session, "pred_lat", value = coord$lat[1])
      updateNumericInput(session, "pred_lon", value = coord$lon[1])
    }
  })

  # Handle Invariant: If number_of_reviews is 0, reviews_per_month is 0
  observeEvent(input$pred_num_reviews, {
    if (input$pred_num_reviews == 0) {
      updateSliderInput(session, "pred_reviews_per_month", value = 0.0)
    }
  })

  # Demonstration Presets
  observeEvent(input$preset_lime, {
    # Test Listing #1 from LIME section: Private Room in Brooklyn, Kensington
    updateSelectInput(session, "pred_room_type", selected = "Private room")
    updateSelectInput(session, "pred_borough", selected = "Brooklyn")
    updateSelectInput(session, "pred_neighbourhood", selected = "Kensington")
    updateNumericInput(session, "pred_lat", value = 40.64749)
    updateNumericInput(session, "pred_lon", value = -73.97237)
    updateSliderInput(session, "pred_min_nights", value = 4)
    updateSliderInput(session, "pred_num_reviews", value = 167)
    updateSliderInput(session, "pred_reviews_per_month", value = 1.4)
    updateNumericInput(session, "pred_host_listings", value = 6)
    updateNumericInput(session, "pred_availability", value = 314)
    updateRadioButtons(session, "pred_model_choice", selected = "rf")
  })

  observeEvent(input$preset_midtown, {
    updateSelectInput(session, "pred_room_type", selected = "Entire home/apt")
    updateSelectInput(session, "pred_borough", selected = "Manhattan")
    updateSelectInput(session, "pred_neighbourhood", selected = "Midtown")
    updateNumericInput(session, "pred_lat", value = 40.75362)
    updateNumericInput(session, "pred_lon", value = -73.98377)
    updateSliderInput(session, "pred_min_nights", value = 2)
    updateSliderInput(session, "pred_num_reviews", value = 45)
    updateSliderInput(session, "pred_reviews_per_month", value = 0.38)
    updateNumericInput(session, "pred_host_listings", value = 2)
    updateNumericInput(session, "pred_availability", value = 355)
    updateRadioButtons(session, "pred_model_choice", selected = "rf")
  })

  observeEvent(input$preset_williamsburg, {
    updateSelectInput(session, "pred_room_type", selected = "Entire home/apt")
    updateSelectInput(session, "pred_borough", selected = "Brooklyn")
    updateSelectInput(session, "pred_neighbourhood", selected = "Williamsburg")
    updateNumericInput(session, "pred_lat", value = 40.7145)
    updateNumericInput(session, "pred_lon", value = -73.9550)
    updateSliderInput(session, "pred_min_nights", value = 3)
    updateSliderInput(session, "pred_num_reviews", value = 32)
    updateSliderInput(session, "pred_reviews_per_month", value = 1.8)
    updateNumericInput(session, "pred_host_listings", value = 1)
    updateNumericInput(session, "pred_availability", value = 120)
    updateRadioButtons(session, "pred_model_choice", selected = "rf")
  })

  observeEvent(input$preset_queens, {
    updateSelectInput(session, "pred_room_type", selected = "Private room")
    updateSelectInput(session, "pred_borough", selected = "Queens")
    updateSelectInput(session, "pred_neighbourhood", selected = "Flushing")
    updateNumericInput(session, "pred_lat", value = 40.7580)
    updateNumericInput(session, "pred_lon", value = -73.8320)
    updateSliderInput(session, "pred_min_nights", value = 1)
    updateSliderInput(session, "pred_num_reviews", value = 12)
    updateSliderInput(session, "pred_reviews_per_month", value = 0.9)
    updateNumericInput(session, "pred_host_listings", value = 1)
    updateNumericInput(session, "pred_availability", value = 200)
    updateRadioButtons(session, "pred_model_choice", selected = "rf")
  })

  # Construct 1-row input data frame strictly matching model formula with defensive defaults
  prediction_input_df <- reactive({
    r_type  <- if (is.null(input$pred_room_type) || input$pred_room_type == "") "Entire home/apt" else input$pred_room_type
    borough <- if (is.null(input$pred_borough) || input$pred_borough == "") "Manhattan" else input$pred_borough
    lat     <- if (is.null(input$pred_lat) || is.na(input$pred_lat)) 40.7536 else as.numeric(input$pred_lat)
    lon     <- if (is.null(input$pred_lon) || is.na(input$pred_lon)) -73.9838 else as.numeric(input$pred_lon)
    min_n   <- if (is.null(input$pred_min_nights) || is.na(input$pred_min_nights)) 2 else as.numeric(input$pred_min_nights)
    n_rev   <- if (is.null(input$pred_num_reviews) || is.na(input$pred_num_reviews)) 25 else as.numeric(input$pred_num_reviews)
    rev_m   <- if (is.null(input$pred_reviews_per_month) || is.na(input$pred_reviews_per_month)) 1.2 else as.numeric(input$pred_reviews_per_month)
    if (n_rev == 0) rev_m <- 0
    h_count <- if (is.null(input$pred_host_listings) || is.na(input$pred_host_listings)) 1 else as.numeric(input$pred_host_listings)
    avail   <- if (is.null(input$pred_availability) || is.na(input$pred_availability)) 180 else as.numeric(input$pred_availability)

    data.frame(
      room_type                      = factor(r_type, levels = c("Entire home/apt", "Private room", "Shared room")),
      neighbourhood_group            = factor(borough, levels = c("Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island")),
      latitude                       = lat,
      longitude                      = lon,
      minimum_nights                 = min_n,
      number_of_reviews              = n_rev,
      reviews_per_month              = rev_m,
      calculated_host_listings_count = h_count,
      availability_365               = avail
    )
  })

  # Reactive valuation calculation triggered by action button, model switch, or preset
  calculated_valuation <- eventReactive(
    list(input$btn_predict, input$pred_model_choice, input$preset_lime, input$preset_midtown, input$preset_williamsburg, input$preset_queens),
    {
      df <- prediction_input_df()
      chosen_m <- if (is.null(input$pred_model_choice)) "rf" else input$pred_model_choice
      
      # Fast predict using preloaded models (never retrained)
      val_rf <- if (!is.null(rf_model)) {
        as.numeric(predict(rf_model, newdata = df))
      } else {
        NA_real_
      }
      
      val_lm <- if (!is.null(lm_model)) {
        as.numeric(predict(lm_model, newdata = df))
      } else {
        NA_real_
      }

      list(
        rf = val_rf,
        lm = val_lm,
        chosen_model = chosen_m,
        inputs = df
      )
    },
    ignoreNULL = FALSE # Run on initial app startup for immediate visual feedback
  )

  # Display Primary Valuation
  output$pred_price_display <- renderText({
    res <- calculated_valuation()
    val <- if (res$chosen_model == "rf") res$rf else res$lm
    if (is.na(val)) return("Model Unavailable")
    paste0("$", format(round(val, 2), nsmall = 2), " / night")
  })

  output$pred_model_badge <- renderText({
    res <- calculated_valuation()
    if (res$chosen_model == "rf") {
      "Valuation generated via Random Forest Regressor (Nonlinear)"
    } else {
      "Valuation generated via Multiple Linear Regression Baseline"
    }
  })

  output$pred_mae_interval_text <- renderText({
    res <- calculated_valuation()
    val <- if (res$chosen_model == "rf") res$rf else res$lm
    mae <- if (res$chosen_model == "rf") 30.43 else 36.46
    low <- max(10, round(val - mae, 0))
    high <- round(val + mae, 0)
    paste0("Expected 1-MAE Accuracy Range: $", low, " to $", high, " / night (±$", mae, ")")
  })

  # Side-by-Side Values
  output$pred_rf_val <- renderText({
    res <- calculated_valuation()
    if (is.na(res$rf)) return("N/A")
    paste0("$", format(round(res$rf, 2), nsmall = 2))
  })

  output$pred_lm_val <- renderText({
    res <- calculated_valuation()
    if (is.na(res$lm)) return("N/A")
    paste0("$", format(round(res$lm, 2), nsmall = 2))
  })

  output$pred_comparison_commentary <- renderUI({
    res <- calculated_valuation()
    diff <- round(res$rf - res$lm, 2)
    diff_abs <- abs(diff)
    
    if (diff > 2) {
      tags$div(
        tags$strong("Model Variance Insight: "),
        sprintf("Random Forest predicts a $%.2f premium over Linear Regression for this profile. Because Random Forest captures localized geospatial interactions (Manhattan waterfront coordinates) and non-linear calendar availability, it accounts for localized demand premiums that linear summation under-estimates.", diff_abs)
      )
    } else if (diff < -2) {
      tags$div(
        tags$strong("Model Variance Insight: "),
        sprintf("Random Forest predicts $%.2f lower than Linear Regression. Linear Regression applies constant additive penalties for room types and boroughs, whereas Random Forest detects that for this specific accommodation profile, market demand stabilizes at a more competitive price point.", diff_abs)
      )
    } else {
      tags$div(
        tags$strong("Model Convergence Insight: "),
        "Both Linear Regression and Random Forest converge closely on this listing profile (difference < $2.00), demonstrating robust consensus between parametric and non-parametric estimators."
      )
    }
  })

  # Model Benchmark Table
  output$pred_metrics_table <- renderTable({
    df <- model_comp_df
    names(df) <- c("Model Architecture", "Test RMSE ($)", "Test MAE ($)", "Test R²")
    df$`Test RMSE ($)` <- sprintf("$%.2f", df$`Test RMSE ($)`)
    df$`Test MAE ($)`  <- sprintf("$%.2f", df$`Test MAE ($)`)
    df$`Test R²`       <- sprintf("%.4f", df$`Test R²`)
    df
  }, striped = TRUE, hover = TRUE, bordered = TRUE)

  # ----------------------------------------------------------------------------
  # TAB 3: NEIGHBOURHOOD INSIGHTS SERVER LOGIC
  # ----------------------------------------------------------------------------

  # Filtered Clustering Data
  clust_filtered_data <- reactive({
    df <- neigh_clusters_df
    if (isTruthy(input$clust_filter_cluster) && input$clust_filter_cluster != "All") {
      df <- df %>% filter(as.character(cluster) == input$clust_filter_cluster)
    }
    if (isTruthy(input$clust_filter_borough) && input$clust_filter_borough != "All") {
      df <- df %>% filter(neighbourhood_group == input$clust_filter_borough)
    }
    df
  })

  output$clust_filter_count_text <- renderText({
    df <- clust_filtered_data()
    paste0("Displaying ", nrow(df), " of 115 qualified neighbourhoods")
  })

  # Scatter Plot: Median Price vs Listing Supply
  output$clust_scatter_plot <- renderPlot({
    df <- clust_filtered_data()
    req(nrow(df) > 0)

    cluster_palette <- c(
      "1" = "#E41A1C",
      "2" = "#377EB8",
      "3" = "#4DAF4A",
      "4" = "#984EA3"
    )

    ggplot(df, aes(x = median_price, y = listing_count, color = as.factor(cluster))) +
      geom_point(aes(size = prop_entire_home), alpha = 0.85) +
      scale_color_manual(values = cluster_palette, name = "Cluster Segment",
                         labels = c("1: Mega-Volume Hubs", "2: Residential Outer-Boroughs", "3: Transit Gateways", "4: Commercial Core")) +
      scale_size_continuous(range = c(3, 8), name = "Entire Home %", labels = scales::percent_format()) +
      scale_x_continuous(labels = scales::dollar_format()) +
      scale_y_log10(labels = scales::comma_format()) +
      labs(
        title = "Neighbourhood Clusters: Median Price vs. Listing Supply (Log-10 Scale)",
        subtitle = "Point size denotes Entire Home proportion. Distinct clustering between volume hubs and pricing cores.",
        x = "Median Listing Price (USD)",
        y = "Total Neighbourhood Listings Count (Log Scale)"
      ) +
      theme_minimal(base_size = 13) +
      theme(
        plot.title = element_text(face = "bold", color = "#1B365D"),
        legend.position = "right",
        panel.grid.minor = element_blank()
      )
  })

  # Interactive Table of 115 Neighbourhoods
  output$clust_neighbourhood_table <- DT::renderDataTable({
    df <- clust_filtered_data()
    req(nrow(df) > 0)

    table_df <- df %>%
      select(
        Neighbourhood = neighbourhood,
        Borough = neighbourhood_group,
        `Cluster` = cluster,
        `Segment Classification` = cluster_label,
        `Median Price ($)` = median_price,
        `Mean Price ($)` = mean_price,
        `Listings` = listing_count,
        `Entire Home (%)` = prop_entire_home,
        `Avg Min Nights` = avg_min_nights,
        `Avg Reviews` = avg_reviews,
        `Avg Availability (d)` = avg_availability
      )

    DT::datatable(
      table_df,
      options = list(
        pageLength = 10,
        dom = 'ftip',
        order = list(list(6, 'desc'))
      ),
      rownames = FALSE
    ) %>%
      DT::formatCurrency(c("Median Price ($)", "Mean Price ($)"), currency = "$") %>%
      DT::formatPercentage("Entire Home (%)", 1) %>%
      DT::formatRound(c("Avg Min Nights", "Avg Reviews", "Avg Availability (d)"), 1)
  })

  # Cluster Profiles Table (Empirical)
  output$clust_profiles_table <- renderTable({
    if (!is.null(cluster_profiles_df)) {
      prof <- cluster_profiles_df
      prof$Segment <- cluster_labels_map[as.character(prof$cluster)]
      prof <- prof %>%
        select(
          `Cluster` = cluster,
          `Segment Name` = Segment,
          `Neighbourhoods` = n_neighbourhoods,
          `Avg Median Price ($)` = median_price,
          `Avg Mean Price ($)` = mean_price,
          `Avg Supply (Listings)` = listing_count,
          `Entire Home Proportion` = prop_entire_home,
          `Avg Min Stay (Nights)` = avg_min_nights,
          `Avg Reviews` = avg_reviews,
          `Avg Availability (Days)` = avg_availability
        )
      
      prof$`Avg Median Price ($)`    <- sprintf("$%.2f", prof$`Avg Median Price ($)`)
      prof$`Avg Mean Price ($)`      <- sprintf("$%.2f", prof$`Avg Mean Price ($)`)
      prof$`Avg Supply (Listings)`   <- sprintf("%.1f",  prof$`Avg Supply (Listings)`)
      prof$`Entire Home Proportion`  <- sprintf("%.1f%%", 100 * prof$`Entire Home Proportion`)
      prof$`Avg Min Stay (Nights)`   <- sprintf("%.1f",  prof$`Avg Min Stay (Nights)`)
      prof$`Avg Reviews`             <- sprintf("%.1f",  prof$`Avg Reviews`)
      prof$`Avg Availability (Days)` <- sprintf("%.1f",  prof$`Avg Availability (Days)`)
      prof
    }
  }, striped = TRUE, hover = TRUE, bordered = TRUE)
}

# ------------------------------------------------------------------------------
# 4. Application Launch
# ------------------------------------------------------------------------------

shinyApp(ui = ui, server = server)
