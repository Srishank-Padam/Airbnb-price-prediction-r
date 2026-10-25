# Airbnb Listing Price Prediction & Neighbourhood Segmentation Using R

An undergraduate Introduction to Data Science (IDS) case study in **R** developed for the B.E. Artificial Intelligence & Data Science curriculum at **CBIT Hyderabad**.

This project establishes an end-to-end, reproducible data science workflow on the New York City Airbnb 2019 dataset (`AB_NYC_2019.csv`). It implements data auditing, defensive cleaning, exploratory data analysis with `ggplot2`, supervised regression modeling (Linear Regression vs. Random Forest), model interpretability using **LIME**, and unsupervised market segmentation using **K-means clustering**.

---

## 1. Project Structure

```
airbnb-ids/
├── AB_NYC_2019.csv        # Primary dataset (Inside Airbnb NYC 2019 snapshot)
├── app.R                  # Interactive R Shiny dashboard for faculty demonstration
├── analysis.R             # Complete, self-contained R script (generates all outputs)
├── report.Rmd             # R Markdown source file
├── report.html            # Knitted standalone HTML report with interactive TOC
├── README.md              # Project documentation and execution guide
└── outputs/
    ├── figures/           # 10 publication-quality PNG figures (300 DPI)
    ├── tables/            # Saved metrics, LIME weights, and cluster profiles (CSV)
    └── models/            # Saved RDS model objects (clean data, lm, rf)
```

---

## 2. Dataset Auditing & Verified Statistics

The analysis is performed strictly on `AB_NYC_2019.csv` without external data augmentation or fabricated statistics:

| Metric / Check | Value / Finding | Interpretation |
| :--- | :--- | :--- |
| **Raw Dimensions** | 48,895 rows, 16 columns | Full snapshot of NYC listings in 2019. |
| **Duplicates** | 0 duplicate rows, 0 duplicate IDs | Every record corresponds to a unique listing ID. |
| **Price Column Class** | `numeric` | Confirmed native numeric type; no currency symbol stripping required. |
| **Raw Price Range** | \$0.00 to \$10,000.00 | Median = \$106.00, Mean = \$152.70, Standard Deviation = \$240.15. |
| **Invalid Prices (\$0)** | 11 rows | Purged (inactive/placeholder accounts). |
| **IQR Price Outliers** | $Q_1 = \$69$, $Q_3 = \$175$, $\text{IQR} = \$106$ | Upper cutoff = $Q_3 + 1.5 \times \text{IQR} = \$334.00$. |
| **Outliers Removed** | 2,972 rows (6.08%) | Removes extreme luxury/joke rates to stabilize regression gradients. |
| **Missing Values** | `reviews_per_month`: 10,052 (20.56%) | Exactly matches `number_of_reviews == 0`; imputed with `0.0`. |
| **Retained Sample** | **45,912 rows** (93.90%) | Final clean sample utilized across all models and clustering. |

---

## 3. Key Findings & Empirical Results

### Supervised Price Prediction (80% Train, 20% Held-Out Test)
Both models were trained on 36,732 observations and evaluated on the exact same 9,180 held-out test observations (`set.seed(123)`):

| Model | Test RMSE | Test MAE | Test $R^2$ | Key Insights |
| :--- | :---: | :---: | :---: | :--- |
| **Multiple Linear Regression** | \$49.23 | \$36.46 | 0.4869 (48.7%) | Private rooms discount by -\$75.26/night; Manhattan adds +\$22.22/night baseline. |
| **Random Forest Regression** | **\$43.53** | **\$30.43** | **0.5988 (59.9%)** | Reduces RMSE by \$5.70, reduces MAE by \$6.03, and explains ~60% of variance. |

### LIME Local Explainability (Test Listing #1)
- **Profile**: Private Room in Brooklyn, 4-night minimum, 167 reviews, 314 days availability.
- **Valuation**: Actual Price = **\$89.00** | Random Forest Prediction = **\$91.01** (error: \$2.01).
- **Local Feature Weights**:
  - `room_type = Private room`: $-\$63.30$ (dominant downward driver vs. entire home).
  - `longitude <= -74.0`: $+\$21.80$ (desirable western transit corridor proximity).
  - `availability_365 > 265`: $+\$11.90$ (commercial year-round listing premium).

### Neighbourhood K-Means Segmentation ($k = 4$, 115 Qualified Neighbourhoods)
Neighbourhoods with $\ge 30$ listings were standardized and clustered based on median price, mean price, listing supply, entire home proportion, minimum stay, review velocity, and availability:
1. **Cluster 1 (Mega-Volume Cultural Hubs; $n = 4$)**: *Bedford-Stuyvesant, Bushwick, Harlem, Williamsburg*. Massive volume (~3,096 listings/neighbourhood), moderate price (\$82.8), dominated by private rooms (38.2% entire homes).
2. **Cluster 2 (Residential Outer-Borough Family Communities; $n = 42$)**: *Astoria, Bay Ridge, Crown Heights, Jackson Heights...* Moderate supply (~267.5 listings), accessible price (\$74.8), family residential orientation.
3. **Cluster 3 (High-Turnover Transit & Airport Gateways; $n = 28$)**: *Allerton, Bayside, Canarsie, Jamaica, Flushing...* Low supply (~91.7 listings), lowest price (\$70.9), shortest minimum stay (3.3 nights), highest review activity (37.3 reviews), highest availability (187.1 days).
4. **Cluster 4 (Premium Commercial Manhattan & Coastal Brooklyn Core; $n = 41$)**: *SoHo, Tribeca, Chelsea, Midtown, West Village, Brooklyn Heights, DUMBO...* Premier median price (**\$150.7**), high supply (~453.7 listings), dominated by full residences (**66.9% entire homes**), longest minimum stays (8.7 nights).

---

## 4. Setup and Execution Instructions

### Prerequisites
- **R (version $\ge$ 4.0)**: Download from [CRAN](https://cran.r-project.org/)
- **RStudio Desktop**: Download from [Posit](https://posit.co/download/rstudio-desktop/)
- **Pandoc**: Automatically bundled with RStudio

### Step 1: Install Required R Packages
Open RStudio and run the following command in the R Console:

```r
install.packages(c(
  "tidyverse",
  "caret",
  "randomForest",
  "lime",
  "cluster",
  "factoextra",
  "tibble",
  "knitr",
  "rmarkdown",
  "shiny",
  "bslib",
  "DT",
  "scales"
))
```

### Step 2: Open the Project & Set Working Directory
In RStudio, navigate to the folder containing `airbnb-ids`:

```r
setwd("path/to/airbnb-ids")
# Or in RStudio: Session -> Set Working Directory -> To Source File Location
```

### Step 3: Execute the Full Analysis Script
Run the complete pipeline from start to finish:

```r
source("analysis.R")
```

This will:
- Inspect and clean the dataset.
- Generate and save all 10 figures in `outputs/figures/`.
- Train and evaluate the models, saving comparison tables in `outputs/tables/`.
- Cache trained models in `outputs/models/`.

### Step 4: Knit the HTML Report
To generate the final interactive HTML report:

- Open `report.Rmd` in RStudio and click the **Knit** button in the editor toolbar, OR
- Run the following command in the R Console:

```r
rmarkdown::render("report.Rmd", output_format = "html_document")
```

The compiled report `report.html` will open in your web browser.

### Step 5: Launch the Interactive R Shiny Application

The project includes `app.R`, a clean academic dashboard built with **Shiny**, **bslib (Bootstrap 5)**, and **ggplot2** designed for faculty demonstration.

#### Option A: From RStudio
1. Open `app.R` in RStudio.
2. Click the **Run App** button in the top-right toolbar of the script editor.

#### Option B: From the R Console
```r
setwd("path/to/airbnb-ids")
shiny::runApp(".", launch.browser = TRUE)
```

#### Option C: From the Terminal / PowerShell
```powershell
Rscript -e "shiny::runApp('.', launch.browser = TRUE)"
```

The interactive dashboard features three core tabs:
- **Tab 1: Overview**: Displays real verified metrics ($N = 45,912$, median: $\$100$, mean: $\$120$, IQR cutoff: $\$334$), interactive borough/room type filtering, price distribution histograms with median markers, and price variation boxplots across room types.
- **Tab 2: Price Prediction**: Real-time valuation inference using pre-trained Linear Regression and Random Forest models (never retrained on button clicks). Includes interactive sliders, dynamic neighbourhood coordinate auto-fill, four faculty demo presets, side-by-side model comparison, and benchmark test metrics ($R^2 = 0.599$, $\text{MAE} = \$30.43$).
- **Tab 3: Neighbourhood Insights**: K-means market segmentation ($k = 4$) across 115 qualified NYC neighbourhoods ($\ge 30$ listings). Includes cluster summary cards, a log-scaled supply vs. price scatter plot, and a searchable, filterable interactive DataTable.

---

## 5. Technical Highlights & Best Practices Demonstrated

1. **Defensive Preprocessing**: Avoids blindly applying currency regex; inspects data types directly; justifies missing imputation via structural column collinearity (`reviews_per_month` vs. `number_of_reviews`).
2. **Leakage Prevention**: Strictly creates the train/test split before fitting and evaluating models; test observations are never exposed during model training or parameter selection.
3. **High-Cardinality Factor Control**: Avoids dumping 221-level raw `neighbourhood` text into regression (which inflates degrees of freedom and causes severe overfitting); leverages it appropriately for unsupervised spatial aggregation.
4. **S3 Method Registration for LIME**: Implements explicit `model_type.randomForest` and `predict_model.randomForest` S3 dispatchers and namespaces `lime::explain` to prevent function collision with `dplyr::explain`.
5. **Reproducibility**: Anchored with `set.seed(123)` across data partitioning, Random Forest bagging, LIME perturbations, and K-means centroid initializations.
