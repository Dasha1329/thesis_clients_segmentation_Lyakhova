# Customer Segmentation: Clustering Analysis Report

**Dataset:** retail e-commerce order data (Q1 2018)
**Notebook:** `client pivot.ipynb`

---

## Table of Contents

1. [Data Loading and Preprocessing](#1-data-loading-and-preprocessing)
2. [Client Feature Matrix Construction](#2-client-feature-matrix-construction)
3. [Outlier Removal](#3-outlier-removal)
4. [Correlation Analysis](#4-correlation-analysis)
5. [Feature Standardization](#5-feature-standardization)
6. [Clustering Methods](#6-clustering-methods)
   - 6.1 [K-means](#61-k-means)
   - 6.2 [Gaussian Mixture Model (EM)](#62-gaussian-mixture-model-em)
   - 6.3 [DBSCAN](#63-dbscan)
   - 6.4 [HDBSCAN](#64-hdbscan)
   - 6.5 [Hierarchical Clustering (Ward Linkage)](#65-hierarchical-clustering-ward-linkage)
7. [RFM Segmentation](#7-rfm-segmentation)
8. [Comparative Summary](#8-comparative-summary)


## 1. Data Assembly and Preprocessing

### 1.1 Objective

The primary objective of this stage was to consolidate raw transactional data from multiple heterogeneous source files into a single, analytically consistent dataset suitable for customer-level modelling. The data covered the period from January 2017 through February 2018 and was originally distributed across bimonthly CSV archives.

### 1.2 Methods

**Source file structure.** The raw data was stored as semicolon-delimited CSV files encoded in Windows-1251 (CP1251), partitioned by pairs of calendar months and named according to the convention `!MM_MM_YY_VSE.csv` (e.g., `!01_02_17_VSE.csv`, `!03&04_17_VSE.csv`). Several files were additionally compressed in RAR archives and required decompression prior to ingestion.

**Filename parsing.** A custom parsing function (`parse_months_year_from_name`) was implemented to programmatically extract the year and month indices from each filename using regular expressions. This approach ensured robust and reproducible identification of the temporal coverage of each file without relying on manual labelling.

**Date filtering and cross-file deduplication.** A key data quality challenge was that records near month boundaries could appear in more than one source file due to the bimonthly partitioning scheme. To address this, each file was filtered to retain only records whose delivery date (`ДатаДоставки`) fell strictly within the calendar months declared in the corresponding filename. This logic was encapsulated in the `load_and_filter` function, which parsed the date column as a datetime object, derived a monthly period, and applied the allowed-period mask before concatenation.

**Dataset construction.** All filtered monthly fragments were concatenated into a single unified dataframe using `pd.concat`. The resulting dataset was persisted in Apache Parquet format (`transactions_2017.parquet`) to ensure efficient storage and fast subsequent reads.

**Delivery filter.** After loading the consolidated Parquet file, an additional filtering step was applied to restrict the analytical dataset to commercially meaningful transactions only. Records were retained exclusively if all three of the following conditions were satisfied simultaneously: the order was not cancelled (`Отменено == 'Нет'`), the order status was "delivered" (`Статус == 'Доставлен'`), and the quantity sold to the client was strictly positive (`КоличествоПроданоКлиенту > 0`). The resulting filtered dataset (`df_delivered`) was sorted by client identifier and transaction date to facilitate time-aware operations in downstream stages.

### 1.3 Results and Conclusions

The assembly pipeline produced a clean, temporally consistent transaction-level dataset spanning 14 months (January 2017 – February 2018). The explicit date-based filtering at the file-loading stage eliminated the risk of duplicated records that would otherwise arise from overlapping monthly partitions. The delivered-only filter ensured that all subsequent analyses were grounded in fulfilled commercial activity, excluding noise introduced by cancellations, returns, and administrative records. The use of the Parquet format at the intermediate stage reduced I/O overhead for all subsequent modelling steps.

---

## 2. Data Loading and Preprocessing

### 2.1 Objective

The primary objective of this stage was to load the raw transactional dataset, enforce correct data types on all relevant columns, and restrict the analytical scope to a well-defined, homogeneous time window. Proper preprocessing at this stage is a prerequisite for constructing a reliable client-level feature matrix in subsequent steps.

### 2.2 Key Methods

**Data ingestion.** The source file (`!01_02-18_VSE.csv`) was stored in Windows-1251 (CP1251) encoding with a semicolon delimiter, which is characteristic of data exports from 1C-based ERP systems commonly used in Russian retail. The file was loaded using `pandas.read_csv` with explicit encoding and separator parameters.

**Numeric type coercion.** Eight financial columns — order amount, document total, unit price, line total, purchase price, margin, service fee, and delivery cost — were stored as strings using the Russian locale convention: a comma as the decimal separator and a non-breaking space (`\xa0`) as a thousands separator. Each column was cleaned by stripping whitespace and non-breaking spaces, replacing commas with periods, and converting to `float64` via `pd.to_numeric` with `errors='coerce'` to handle any residual malformed entries gracefully.

**Datetime parsing.** Three temporal columns were converted to `datetime64` objects: `Дата` (order creation timestamp, with time-of-day precision), `ДатаДоставки` (actual delivery date), and `ДатаЗаказаНаСайте` (website order placement date). Parsing was performed with `dayfirst=True` to match the DD.MM.YYYY format used in the source system.

**Categorical normalisation.** The `МетодДоставки` (delivery method) and `ФормаОплаты` (payment method) columns were normalised to lowercase with leading/trailing whitespace removed, eliminating case-sensitivity inconsistencies introduced during manual data entry.

**Temporal filtering.** Prior to type casting the delivery date range spanned from 24 February 2017 to 22 February 2019 — a window of over two years containing data from multiple business periods with potentially different seasonality, pricing policies, and product assortments. To ensure analytical comparability, the dataset was restricted to orders with a delivery date falling within Q1 2018 (1 January – 31 March 2018 inclusive).

### 2.3 Results and Findings

After temporal filtering the working dataset comprised **692,856 transaction rows** across **38 columns**. The columns can be grouped into the following thematic blocks:

| Category | Columns |
|---|---|
| Temporal | `Дата`, `ДатаДоставки`, `ДатаЗаказаНаСайте`, `МесяцДатыЗаказа`, `ГодДатыЗаказа` |
| Order identifiers | `НомерЗаказаНаСайте`, `НомерСтроки`, `НовыйСтатус`, `Статус`, `Отменено`, `ПричинаОтмены` |
| Financial | `СуммаЗаказаНаСайте`, `СуммаДокумента`, `Цена`, `СуммаСтроки`, `ЦенаЗакупки`, `Маржа`, `СуммаУслуг`, `СуммаДоставки` |
| Product | `Группа2`, `Группа3`, `Группа4`, `Тип`, `Номенклатура`, `ТипТовара`, `ID_SKU`, `Количество`, `КоличествоПроданоКлиенту` |
| Logistics | `МетодДоставки`, `ПВЗ_код`, `МагазинЗаказа`, `ГородМагазина` |
| Geography | `Регион`, `Гео` |
| Client | `Телефон_new`, `ЭлектроннаяПочта_new`, `Клиент` |

The monthly distribution of delivered orders within the filtered window was as follows:

| Month | Delivered rows |
|---|---|
| January 2018 | 215,839 |
| February 2018 | 375,489 |
| March 2018 | 101,528 |
| **Total** | **692,856** |

The pronounced spike in February is consistent with typical post-holiday demand patterns observed in Russian e-commerce, as well as potential campaign-driven sales activity around Valentine's Day and the Defender of the Fatherland Day (23 February).

Inspection of the three date columns (`Дата`, `ДатаДоставки`, `ДатаЗаказаНаСайте`) confirmed that all three are populated throughout the dataset. The presence of multiple timestamps per order enables calculation of delivery lead times and identification of temporal anomalies, such as cases where the recorded delivery date precedes the website order date — a data quality issue flagged for downstream handling.

### 2.4 Visualisation

A bar chart of monthly delivery counts across the three months of Q1 2018 visually confirms the uneven distribution: February accounts for approximately 54% of all transactions in the window, January for 31%, and March for the remaining 15%. This imbalance is an inherent property of the data and is not introduced by the filtering procedure; it reflects the actual operational rhythm of the business during this period.

---

## 3. Client–Quarter Feature Matrix Construction

### 3.1 Objective

The objective of this stage was to transform the transaction-level dataset into a structured analytical panel in which each observation represents a unique combination of a client and a calendar quarter. This client–quarter matrix serves as the foundational input for all subsequent segmentation and predictive modelling steps.

### 3.2 Methods

**Base aggregation.** The delivered transactions were grouped by client identifier (`Телефон_new`) and calendar quarter (`quarter`), yielding the following core metrics per client per quarter: number of distinct delivered orders (`delivered_orders`), total items purchased (`items`), total revenue (`revenue`), total margin (`margin`), dates of the first and last purchase within the quarter, and average order value (`avg_order_value = revenue / delivered_orders`).

**Temporal features.** Two time-aware features were computed to capture the recency and tenure dimensions of client behaviour:
- *Recency* (`recency_days`): the number of days elapsed between the client's most recent purchase prior to or on the last day of the quarter and the quarter-end date. This was computed using a time-ordered backward `merge_asof` join, which correctly handles the absence of purchases in a given quarter by looking back to prior quarters.
- *Tenure* (`days_since_first_purchase`): the number of days between the client's globally earliest purchase date (across all quarters) and the quarter-end date. Client–quarter rows predating the client's first purchase were excluded to avoid spurious recency values.

**Modal order size.** For each client–quarter, the modal number of items per order (`most_freq_order_size`) was derived by first computing order-level item totals and then applying a mode aggregation. This feature characterises the client's typical basket size within a given quarter.

**Product category shares.** The share of items purchased from each product category (`Группа2`) relative to the client's total quarterly item volume was computed and pivoted into wide format, yielding one column per category. The dataset contained 13 distinct product categories, including children's food, toys, cosmetics, diapers, footwear, home appliances, and pet goods, among others.

**Regional shares.** An analogous pivot was constructed for geographic regions (`Регион`), representing the fraction of items delivered to each of seven macro-regions: Central, Far East, North, Volga (Privolzie), Siberia, Southern, and Ural.

**Delivery method shares.** The share of orders placed via each fulfilment method (`МетодДоставки`) — pick-up point, courier, retail store, and self-pickup — was computed relative to the total number of delivered orders per client–quarter.

**Payment method shares.** Similarly, the proportion of orders paid by cashless versus cash means (`ФормаОплаты`) was computed and appended to the matrix.

**Cancellation and return rates.** Drawing from the full (unfiltered) transaction dataset, three additional order quality metrics were merged: the total order count per client–quarter (`total_orders`), the cancellation rate (`cancel_ratio = cancelled orders / total orders`), the return rate (`return_ratio = returned orders / total orders`), and the delivery success rate (`delivered_ratio = delivered orders / total orders`).

**Final schema.** The resulting client–quarter matrix comprised approximately 44 features grouped into the following thematic blocks: identifiers (client, quarter), order volume and fulfilment quality, financial metrics, lifecycle / temporal metrics, product category composition, regional distribution, delivery preferences, and payment preferences.

### 3.3 Results and Conclusions

The feature engineering pipeline successfully produced a rich, multidimensional panel dataset in which each row encodes a client's behavioural and transactional profile for a specific quarter. The combination of financial aggregates, temporal recency and tenure signals, categorical composition shares, and order quality ratios provides a comprehensive representation of customer behaviour at the quarterly level. This matrix constitutes the direct input to the RFM segmentation and all machine learning models constructed in subsequent stages of the analysis.

---

## 4. Outlier Removal and Feature Selection

### 4.1 Objective

Before applying clustering algorithms, the client feature matrix required two additional preparatory steps: removal of extreme outliers that could distort distance-based computations, and elimination of redundant features identified through pairwise correlation analysis. Both steps aim to ensure numerical stability and reduce the risk of spurious cluster structure driven by a small number of highly atypical observations or by linear dependencies among features.

### 4.2 Outlier Removal

**Method.** Outlier detection was applied to three key financial features: `revenue`, `margin`, and `avg_order_value`. These variables exhibit strongly right-skewed distributions with heavy tails — a characteristic pattern in e-commerce client data, where the vast majority of customers generate modest revenue while a small fraction generates disproportionately large volumes.

A robust, modified IQR-based procedure was employed. Rather than using the standard first and third quartiles (Q1 = 25th percentile, Q3 = 75th percentile), the bounds were computed from the 10th and 90th percentiles, yielding a wider reference range that is less sensitive to the distribution tails. The outlier threshold was further expanded by a factor of 3 (compared to the conventional factor of 1.5), so that only genuinely extreme values — far beyond the empirical spread of the central 80% of clients — were flagged:

$$\text{lower bound}_k = Q_{0.1}(X_k) - 3 \cdot \text{IQR}_{[0.1, 0.9]}(X_k)$$
$$\text{upper bound}_k = Q_{0.9}(X_k) + 3 \cdot \text{IQR}_{[0.1, 0.9]}(X_k)$$

A client was classified as an outlier only if they violated the bounds on **at least two out of three** financial features simultaneously. This multi-condition criterion substantially reduces the risk of discarding economically legitimate high-value customers who may be extreme on a single dimension (e.g., unusually high average order value) but otherwise plausible. It focuses removal on clients whose entire financial profile is anomalous.

**Result.** A total of **656 clients** were identified and removed, reducing the dataset from 83,059 to **82,403 clients**. These records were characterised by extreme financial metrics across multiple dimensions and would have distorted the scale and shape of the feature distributions, thereby negatively affecting the performance of distance-based and density-based clustering algorithms.

### 4.3 Correlation Analysis and Feature Reduction

**Method.** A pairwise Pearson correlation matrix was computed over all numeric features of the post-removal client matrix. Feature pairs with |r| > 0.75 were flagged as potentially redundant. A correlation heatmap provided a visual overview of the inter-feature dependency structure.

**Findings.** Eight highly correlated pairs were identified:

| Feature A | Feature B | r |
|---|---|---|
| `revenue` | `avg_order_value` | +0.857 |
| `recency_days` | `days_since_first_purchase` | +0.837 |
| `delivery_курьерская` | `delivery_магазины` | −0.834 |
| `delivery_курьерская` | `payment_безналичная` | −0.947 |
| `delivery_курьерская` | `payment_наличная` | +0.947 |
| `delivery_магазины` | `payment_безналичная` | +0.881 |
| `delivery_магазины` | `payment_наличная` | −0.881 |
| `payment_безналичная` | `payment_наличная` | −1.000 |

The perfect anti-correlation between `payment_безналичная` and `payment_наличная` (r = −1.000) is a mathematical consequence of their construction as complementary proportions summing to 1, making one column entirely redundant. More broadly, the payment method shares were found to be near-perfectly collinear with the delivery method shares: courier delivery is almost exclusively associated with card payment, while in-store pickup correlates strongly with cash. This structural dependency means that payment features carry no additional information beyond what is already encoded in the delivery method shares.

**Columns removed.** Based on this analysis, four columns were dropped from the feature matrix:

- `delivery_магазины` — strongly anti-correlated with `delivery_курьерская` and redundant given that the remaining delivery method shares are mutually informative
- `days_since_first_purchase` — correlated with `recency_days` (r = 0.837); within a single-quarter observation window, tenure and recency largely coincide
- `payment_безналичная` — informationally redundant with delivery method features
- `payment_наличная` — perfectly anti-correlated with `payment_безналичная`

**Final feature matrix.** After outlier removal and feature reduction, the client matrix comprised **82,403 rows × 37 columns** (including the client identifier). The 36 analytical features entering downstream clustering models span behavioural, financial, temporal, product preference, geographic, and logistics dimensions, with all major sources of multicollinearity resolved.

### 4.4 Data Export

The cleaned and finalised client feature matrix was saved to `data/client_pivot.csv` for reproducible use across subsequent modelling stages. The export step was implemented with explicit file-overwrite logic to prevent accidental accumulation of stale versions.

---

## 5. Correlation Analysis

*— to be written —*

---

## 6. Feature Standardization

### 6.1 Objective

Clustering algorithms based on distance metrics — such as K-means, DBSCAN, and Gaussian Mixture Models — are sensitive to the absolute scale and distributional shape of input features. Features measured in large monetary units (e.g., revenue in rubles) will dominate Euclidean distances and effectively suppress the contribution of bounded proportional features (e.g., category shares in the [0, 1] range) unless proper normalisation is applied. The objective of this stage was to transform all features into a common numerical scale while accounting for the heterogeneous distributional properties of different feature groups.

### 6.2 Preparatory Steps

**Data reload and ID removal.** The cleaned client matrix was reloaded from `data/client_pivot.csv` (82,403 rows × 37 columns). The client identifier column (`Телефон_new`) was dropped prior to scaling, leaving 36 purely analytical features.

**Manual exclusion of invalid records.** Four client records identified as technically invalid through manual inspection (phone number hashes corresponding to test or system accounts) were removed, resulting in a final analytical dataset of **82,399 clients × 36 features**.

**Distribution inspection.** For all 36 features, paired histogram and box-plot visualisations were generated to assess distributional shape, skewness, and the presence of residual extreme values. This inspection informed the differentiated treatment described below.

### 6.3 Transformation Strategy

The 36 features were partitioned into three groups based on their distributional properties, each receiving a distinct transformation pipeline.

**Group 1 — Heavy-tailed count and financial features** (`log1p` transformation + `RobustScaler`):

`items`, `revenue`, `margin`, `avg_order_value`, `most_freq_order_size`

These variables exhibit strong right-skewed distributions with long tails — a typical characteristic of transactional financial data. Two operations were applied sequentially:

1. *Shift and log-transform.* The `margin` feature contains negative values (clients with a net loss over the period), so it was first shifted by subtracting its minimum and adding 1 to ensure all values are strictly positive. All five features were then transformed via $x' = \log(1 + x)$, which compresses large values while preserving the ordering and interpretability of the data. This transformation substantially reduces skewness and brings the distributions closer to symmetry.

2. *RobustScaler.* After log-transformation, the features were standardised using `RobustScaler`, which centres each feature at its median and scales by the interquartile range (IQR) rather than the mean and standard deviation. This approach is robust to residual outliers that survive the log-transformation, as it does not penalise extreme values during the scaling computation.

**Group 2 — Bounded ratio features** (`StandardScaler` only):

`delivered_ratio`, `cancel_ratio`, `return_ratio`, `recency_days`

These features either represent proportions naturally constrained to [0, 1] or have distributions that are sufficiently well-behaved (e.g., approximately unimodal) to admit direct z-score standardisation. `StandardScaler` centres each feature at its mean and scales by its standard deviation, producing zero-mean, unit-variance representations.

**Group 3 — Proportion-based categorical share features** (`StandardScaler`):

All remaining 27 columns: `total_orders`, 13 product category share columns, 8 regional share columns, and 5 delivery method share columns.

These features are already expressed as proportions or low-count integers and do not exhibit the extreme tail behaviour of financial variables. Standard z-score normalisation is applied uniformly across this group.

### 6.4 Results

The output of the standardisation pipeline is a fully numerical matrix `df_scaled` of shape **(82,399 × 36)**, where all features are expressed on comparable scales. The transformation choices reflect a principled trade-off: log-compression reduces the disproportionate influence of high-revenue clients on distance computations, while the use of `RobustScaler` for the most volatile features ensures that any residual extreme observations do not distort the scaling parameters. The resulting feature space provides a stable and balanced representation of client behaviour suitable for all downstream clustering methods.

---

## 7. Clustering Methods

### 7.1 K-means

#### 6.1.1 Objective

K-means is the baseline partitioning method in this study. Its objective is to partition the client feature space into *k* non-overlapping, spherically shaped clusters by minimising the total within-cluster sum of squared distances (inertia) to cluster centroids. Despite its well-known sensitivity to initialisation and its assumption of isotropic clusters, K-means provides a computationally efficient and interpretable segmentation that serves as a reference point for evaluating the more complex methods applied subsequently.

#### 6.1.2 Selecting the Optimal Number of Clusters

Selecting the number of clusters *k* is a critical and inherently ill-posed problem, as no single criterion is universally optimal. A comprehensive evaluation was conducted over the range k ∈ {5, …, 21}, computing four complementary internal validation metrics for each value of *k*:

- **SSE (inertia / Elbow method):** measures total within-cluster variance; lower is better. The "elbow point" — the value of *k* beyond which the marginal reduction in SSE diminishes sharply — provides a heuristic upper bound on the informativeness of additional clusters.
- **Silhouette Score:** measures the ratio of mean intra-cluster distance to mean nearest-cluster distance for each point, averaged over all points. Values range from −1 to +1; higher values indicate better-defined, well-separated clusters.
- **Davies–Bouldin Index (DBI):** measures the average similarity between each cluster and its most similar neighbour, defined as the ratio of intra-cluster scatter to inter-cluster separation. Lower values indicate better separation.
- **Calinski–Harabasz Index (CH):** the ratio of between-cluster to within-cluster dispersion. Higher values indicate denser, more separated clusters.

All four metrics were computed with `n_init=10` to mitigate sensitivity to initialisation. Additionally, a Gap Statistic was computed by comparing the observed log-inertia against the expected log-inertia of uniformly distributed reference datasets sampled within the bounding box of the scaled feature space, averaged over B = 10 reference samples.

The metrics were visualised both individually (six-panel plot) and jointly after min-max normalisation to a common [0, 1] scale (with SSE and DBI inverted so that higher always means better), enabling a unified visual comparison across all criteria. The converging signal from the multi-metric analysis indicated candidate values at **k = 10, 11, and 15**. A final model was fitted with **k = 10**, balancing segmentation granularity against cluster stability and interpretability.

#### 6.1.3 Model Fitting

The K-means model was fitted on the standardised feature matrix `df_scaled` (82,399 × 36) with the following configuration:

- `n_clusters = 10`
- `random_state = 42` (for reproducibility)
- `n_init = 10` (ten independent random initialisations; best result retained)

Cluster labels were assigned to each client and appended to the unscaled client matrix as column `k_means_10`.

#### 6.1.4 Internal Validation Metrics

| Metric | Value |
|---|---|
| Silhouette Score | 0.148 |
| Davies–Bouldin Index | 1.982 |
| Calinski–Harabasz Index | 3,766.7 |

The relatively low Silhouette Score (0.148) is consistent with expectations for a high-dimensional dataset with 36 features, where the concentration of measure phenomenon causes distances between all pairs of points to become increasingly similar, compressing the range of achievable silhouette values. In such settings, absolute metric thresholds developed for low-dimensional data are not directly applicable; the metrics are more informative as comparative measures across methods than as absolute quality indicators.

#### 6.1.5 Cluster Structure and Distribution

The ten clusters ranged considerably in size, reflecting genuine heterogeneity in the client population:

| Cluster | Client count | Share |
|---|---|---|
| 0 | 4,788 | 5.8% |
| 1 | 7,103 | 8.6% |
| 2 | 25,360 | 30.8% |
| 3 | 3,617 | 4.4% |
| 4 | 1,448 | 1.8% |
| 5 | 12,130 | 14.7% |
| 6 | 2,800 | 3.4% |
| 7 | 6,153 | 7.5% |
| 8 | 9,819 | 11.9% |
| 9 | 9,181 | 11.1% |

Cluster 2 is by far the largest, accounting for nearly one-third of all clients, while Cluster 4 is the smallest (1.8%). This asymmetry is typical of real-world customer segmentation tasks where a large segment of infrequent, low-value buyers coexists with several smaller, more behaviourally distinct groups.

The median cluster profiles on key financial and behavioural features reveal clear differentiation:

| Cluster | Median orders | Median revenue | Median margin | Median AOV | Median recency | Courier share |
|---|---|---|---|---|---|---|
| 0 | 1 | 2,237 | 255.6 | 1,942 | 19 days | 25% |
| 1 | 1 | 4,243 | 265.5 | 3,115 | 19 days | 49% |
| 2 | 1 | 1,691 | 256.3 | 1,589 | 19 days | 16% |
| 3 | 3 | 2,575 | 336.3 | 1,871 | 17 days | 4% |
| 4 | 1 | 2,097 | 76.1 | 1,898 | 17 days | 20% |
| 5 | 1 | 2,864 | 449.5 | 2,418 | 22 days | 15% |
| 6 | 1 | 2,398 | 305.2 | 2,119 | 16 days | 0% |
| 7 | 1 | 2,099 | 271.4 | 1,950 | 20 days | 0% |
| 8 | 1 | 2,289 | 63.3 | 2,089 | 19 days | 14% |
| 9 | 1 | 4,887 | 1,110 | 4,500 | 17 days | 15% |

Cluster 9 stands out as a premium segment: the highest median revenue (4,887), the highest median margin (1,110), and the highest average order value (4,500), with a comparatively low cancel ratio (5%). Clusters 4 and 8 are characterised by near-zero median margins (76 and 63 respectively), suggesting that the profitability of purchases in these segments is very low despite non-trivial revenue figures. Cluster 3 is the only group with a median order count of 3, indicating a more loyal repeat-purchase profile. Clusters 6 and 7 have zero courier delivery share, concentrated entirely on in-store pickup (магазины) and СДЭК respectively — a geographical or behavioural signal that clearly differentiates them from courier-dominant segments.

#### 6.1.6 Visualisations

**PCA projection.** The 36-dimensional scaled feature space was reduced to two principal components using PCA for visual inspection. The scatter plot of clients coloured by cluster assignment reveals that while the clusters are not sharply separated in this 2D projection — which is expected given that PCA retains only a fraction of total variance — distinct spatial concentrations are visible, confirming that the K-means solution has identified non-trivial structure.

**Pie chart.** The distribution of clients across the ten clusters was visualised as a pie chart, highlighting the dominance of Cluster 2 and the small sizes of Clusters 3 and 4.

**Violin plots.** Distributions of log-transformed revenue and margin across clusters were visualised using violin plots. The plots confirm that clusters differ not only in central tendency but also in the shape and spread of their financial distributions: Cluster 9 shows a narrow, high-value distribution, while Cluster 2 exhibits a broad, low-value spread consistent with a heterogeneous mass-market segment.

#### 6.1.7 Business Interpretation of Segments

The cluster profiles carry direct operational implications, summarised below by segment:

**Cluster 0 — Hygiene & diapers, moderate profitability.** Clients purchasing primarily cosmetics/hygiene and diapers with a typical basket of ~4–6 items and moderate AOV (~1,942 RUB). A recurring, predictable demand pattern well-suited to retention via subscription mechanics, "hygiene + diapers" bundles, and frequency-triggered promotions.

**Cluster 1 — Large baby-food baskets, margin at risk.** High volume (items = 16, AOV ~3,115 RUB) but the highest cancel ratio across all segments (0.18) and low margin (~6%). This "wholesale baby-food" profile suggests price-sensitive buyers vulnerable to out-of-stock and substitution issues. The segment requires root-cause analysis of both low margin (discounts? delivery costs?) and cancellation drivers before scaling marketing spend.

**Cluster 2 — Single-item toys & footwear, healthy margin.** Predominantly one-item purchases (items = 1) with high margin (~15%) and good service quality. A strong candidate for frequency uplift via age/occasion-triggered recommendations and cross-sell into adjacent gift categories.

**Cluster 3 — Service red zone: returns and non-deliveries.** Delivered ratio ~0.47 and return ratio ~0.41 — the most operationally distressed segment. Revenue and margin are mid-range, but the business reality is direct financial loss and customer trust damage. The priority here is operational diagnostics (product quality, sizing, packaging, carrier or warehouse issues), not marketing. Fixing the root cause must precede any retention effort.

**Cluster 4 — Pet products: high volume, near-zero margin.** Near-pure pet goods (share ~0.94) with very low margin (~3.6%) despite reasonable revenue. A commodity segment that risks consuming operational capacity without profit return. Strategy options: margin improvement via private label or curated assortment, controlled promotion spend, or use as a cross-sell entry point into higher-margin categories.

**Cluster 5 — Textiles, high margin, cooling off.** Almost exclusively textiles/knitwear (share ~0.86) with the highest margin among all segments (~15.7%), but the highest recency (22 days) — these clients are drifting away. A straightforward reactivation opportunity: seasonal triggers, new collection alerts, size/age-personalised selections, and "second item" discount mechanics.

**Cluster 6 — Siberia: large goods & toys, logistics gap.** Geographic concentration in SIBERIA (~100%) with toys and large-size goods. Delivery ratio (0.88) is below the segment average — a logistics rather than demand problem. Improving regional SLA, carrier selection, and packaging for bulky items would directly raise retention for this group.

**Cluster 7 — North: toys & textiles, stable profitability.** Geographically concentrated in NORTH (~100%), healthy margins (~12.9%), and no red flags on service quality. A well-functioning segment suitable for assortment expansion and frequency growth via proven best practices.

**Cluster 8 — Diapers: structural low margin, subscription potential.** Near-pure diapers (share ~0.89) with the lowest margin (~2.8%) in the solution. The commercial rationale for retaining this segment lies not in diaper margin but in lifetime value through cross-sell (wipes, baby care, baby food, household) and subscription/auto-repeat mechanics that convert a commodity purchase into an ongoing relationship.

**Cluster 9 — Premium large goods: the profit star.** Near-pure large-size goods (share ~0.89), highest revenue (~4,887 RUB), highest margin (~22.7%), and highest AOV (~4,500 RUB) in a single-item, high-precision purchase pattern. Service quality is excellent. This is the segment to protect and scale carefully: priority support, best-in-class delivery experience, accessory/consumable upsell, and post-purchase communication to encourage repeat or category extension.

---

### 7.2 Gaussian Mixture Model (EM)

#### 6.2.1 Objective

The Gaussian Mixture Model (GMM) with Expectation-Maximisation (EM) fitting extends the hard cluster assignments of K-means into a fully probabilistic framework. Each cluster is modelled as a multivariate Gaussian distribution with its own mean vector and covariance matrix, and clients are assigned to clusters according to posterior membership probabilities rather than hard boundaries. This allows the model to represent clusters of varying shape, density, and orientation — a more flexible assumption than the spherical cluster geometry implicitly required by K-means. The soft-assignment mechanism also enables assessment of classification certainty for each individual client.

#### 6.2.2 Selecting the Number of Components

Two rounds of model selection were conducted, each sweeping over a range of *k* values:

**Round 1** (k ∈ {5, …, 34}, `covariance_type='full'`, `n_init=5`): BIC, AIC, and average log-likelihood were computed for each model. Full-covariance GMMs are the most expressive variant (each component has its own unconstrained covariance matrix) and the most computationally expensive. This sweep ran for approximately 8 hours. The initial BIC curve indicated a candidate near **k = 18**.

**Round 2** (k ∈ {2, …, 20}, identical configuration): A finer-grained sweep confirmed that BIC continued declining up to the boundary of the search range, reaching its minimum at **k = 20**. In parallel, the AIC curve — which penalises model complexity less aggressively than BIC — pointed to an even larger number of components. However, inspection of pie-chart grids showing the client distribution across clusters at each *k* revealed that solutions above k ≈ 11 produced multiple near-empty clusters (degenerate components absorbing only a handful of observations), which would not be practically interpretable.

Balancing statistical criteria against interpretability and cluster stability, **k = 11** was selected as the final number of components. BIC and AIC are designed for parametric model comparison and tend to favour increasing complexity when the true data-generating process is not a finite Gaussian mixture — a known limitation in applied segmentation contexts.

**Model selection criteria:**

- **BIC (Bayesian Information Criterion):** $\text{BIC} = -2 \ln \hat{L} + p \ln n$, where $\hat{L}$ is the maximised likelihood, $p$ is the number of free parameters, and $n$ is the sample size. BIC applies a stronger complexity penalty than AIC, making it the preferred criterion when parsimony is valued.
- **AIC (Akaike Information Criterion):** $\text{AIC} = -2 \ln \hat{L} + 2p$. Lower is better in both cases.
- **Average log-likelihood:** the mean per-sample log-likelihood under the fitted model; higher indicates a better fit.

#### 6.2.3 Model Fitting

The final GMM was fitted with the following configuration:

- `n_components = 11`
- `covariance_type = 'full'` — each component has a full, unconstrained covariance matrix, permitting clusters of arbitrary ellipsoidal shape
- `random_state = 42`
- `n_init = 5` — five independent EM initialisations; the best result by log-likelihood was retained
- `max_iter = 500`, `tol = 1e-3`

Hard cluster assignments were obtained via `predict()` (MAP label); soft probabilistic assignments via `predict_proba()` (posterior membership probabilities over all 11 components).

#### 6.2.4 Internal Validation Metrics

| Metric | Value |
|---|---|
| Silhouette Score | 0.087 |
| Davies–Bouldin Index | 2.960 |
| Calinski–Harabasz Index | 2,363.6 |

Compared to K-means (k = 10), the GMM solution shows lower scores across all three metrics. This is partly expected: GMM clusters are not constrained to minimise Euclidean within-cluster variance, so metrics based on Euclidean geometry (silhouette, DBI, CH) are less directly aligned with the model's optimisation objective. The GMM is optimising log-likelihood under a Gaussian model, not geometric compactness. These metrics should therefore be interpreted as comparative reference points rather than absolute quality measures.

**Assignment certainty.** The maximum posterior probability across the 11 components was computed for each client as a certainty measure. All 82,399 clients had a certainty score above 0.5 (i.e., zero clients were ambiguously assigned), indicating that despite the low Silhouette Score, the model produced confident, non-overlapping assignments for every observation.

#### 6.2.5 Cluster Structure and Profiles

The eleven clusters exhibit substantially greater behavioural differentiation than the K-means solution, owing to the model's capacity to capture elongated and non-spherical cluster geometries. The mean cluster profiles reveal distinct segments:

| Cluster | Avg orders | Avg revenue | Avg margin | Avg AOV | Dominant category | Delivery | Payment |
|---|---|---|---|---|---|---|---|
| 0 | 2.18 | 6,420 | 1,193 | 3,034 | Toys (19%), Textiles (22%) | Mixed | 71% card |
| 1 | 1.00 | 2,244 | 402 | 2,244 | Stationery/Books (37%), Pet products (20%) | СДЭК (8%) | 100% card |
| 2 | 1.33 | 3,205 | 589 | 2,454 | Footwear (61%) | In-store | 100% card |
| 3 | 1.00 | 4,058 | 821 | 4,058 | Toys (43%) | 100% courier | 0% card (cash) |
| 4 | 1.56 | 4,716 | 987 | 3,035 | Large goods (25%), Toys (23%) | Self-pickup (26%) | 74% card |
| 5 | 1.00 | 2,736 | 514 | 2,736 | Toys (46%), Textiles (25%) | 100% pick point | 100% card |
| 6 | 1.86 | 6,111 | 1,015 | 3,531 | Footwear (20%), Pet products (13%) | 70% courier | 30% card |
| 7 | 2.03 | 7,466 | 1,163 | 3,853 | Textiles (17%), Hygiene (16%) | Mixed + СДЭК (11%) | 75% card |
| 8 | 1.51 | 4,451 | 636 | 2,982 | Feeding products (17%), Baby food (29%), Hygiene (19%) | Mixed | 77% card |
| 9 | 1.00 | 2,021 | 391 | 2,021 | Toys (99%) | СДЭК (4%) | 100% card |
| 10 | 1.00 | 4,397 | 872 | 4,397 | Large goods (37%), Textiles (40%) | In-store | 100% card |

Several segments stand out:

- **Cluster 9** is a near-pure single-category segment (99% toys), with low revenue (2,021) but zero cancel and return rates — likely parents making deliberate, carefully chosen toy purchases.
- **Cluster 3** is the only entirely cash-paying segment (0% card), exclusively using courier delivery — a behavioural combination atypical of the broader client base and potentially indicative of a specific demographic or geographic cohort.
- **Cluster 5** uses pick-point delivery exclusively (100%) — a logistics preference that sharply distinguishes this group.
- **Cluster 7** has the highest average revenue (7,466) and margin (1,163) among all segments — the top-value group in the GMM solution.
- **Cluster 4** exhibits the highest return ratio (21%) alongside a high cancel ratio (9%), making it the highest-attrition segment in the solution.
- **Cluster 2** is dominated by footwear (61%) — a high-return-rate category by nature — yet shows low return rates within this cluster (0%), suggesting that the footwear buyers here are satisfied and deliberate purchasers.

#### 6.2.6 Visualisations

**AIC/BIC curves.** The model selection process was visualised as dual line charts of BIC and AIC against *k* (range 2–20). Both curves decline monotonically, with BIC exhibiting a slight inflection point around k = 5–9 before resuming its decline.

**Pie-chart grid.** A 4×4 grid of pie charts (k = 5 to 20) shows the evolving distribution of clients across components as *k* increases, revealing the emergence of degenerate near-empty clusters for large *k* values.

**Box plots.** For all 30 features, per-cluster box plots were generated using log-transformed scales for count and financial variables and raw scales for proportion features. These plots reveal that the GMM segments differ substantially on category preference shares, delivery method usage, and recency — beyond the financial differentiation visible in K-means.

---

### 7.3 DBSCAN

#### 6.3.1 Objective

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a non-parametric, density-based clustering algorithm that does not require the number of clusters to be specified in advance. Unlike K-means and GMM, DBSCAN identifies clusters as contiguous regions of high point density separated by low-density regions, and explicitly labels low-density points as noise (label −1). This makes it particularly well-suited to discovering clusters of irregular shape and to separating genuine structural groups from sparse, atypical observations — an important property for customer data that may contain isolated behavioural outliers.

#### 6.3.2 Dimensionality Reduction via PCA

DBSCAN relies on pairwise Euclidean distances, which become increasingly uninformative in high-dimensional spaces due to the concentration of measure phenomenon (the "curse of dimensionality"). To mitigate this, PCA was applied to the 36-dimensional scaled feature matrix prior to DBSCAN fitting.

A cumulative explained-variance plot was examined to identify the number of principal components needed to retain a substantial portion of variance. **25 components** were selected, retaining **89.96%** of total variance in the original feature space. This reduces dimensionality by 31% while preserving the vast majority of the signal, making distance computations in the PCA subspace more geometrically meaningful.

#### 6.3.3 Parameter Selection: `eps` and `min_samples`

DBSCAN has two key hyperparameters:

- **`eps` (ε):** the neighbourhood radius; a point is considered a core point if at least `min_samples` other points lie within distance ε.
- **`min_samples`:** the minimum number of points in a neighbourhood for a point to be classified as a core point.

**`min_samples` selection.** A value of `min_samples = 70` was used, following the heuristic of approximately 2–3 × the number of PCA dimensions (25), reflecting the expectation that meaningful dense regions should contain a non-trivial number of clients.

**`eps` selection via k-distance graph.** The distances to the 20th nearest neighbour were computed for all points in PCA space and sorted in ascending order (k-distance graph). The graph displays a characteristic "elbow" at the transition from the dense core of the data to the sparse periphery. The 90th–99th percentile range of these distances was isolated for detailed examination. Based on the elbow location, three candidate values — ε ∈ {2.0, 2.5, 3.0} — were evaluated in a grid search:

| ε | Clusters found | Noise share | Largest cluster share |
|---|---|---|---|
| 2.0 | 25 | 17.4% | 58.4% |
| 2.5 | 20 | 9.6% | 61.8% |
| 3.0 | 15 | 6.1% | 70.8% |

**ε = 2.5** was selected as the final parameter, offering the best balance between noise suppression (9.6%) and cluster resolution (20 clusters), avoiding both the fragmentation of ε = 2.0 and the over-merging of ε = 3.0.

#### 6.3.4 Model Fitting

The final DBSCAN model was fitted on the 25-component PCA representation with:

- `eps = 2.5`
- `min_samples = 70`
- `metric = 'euclidean'`
- `n_jobs = -1` (parallel computation)

#### 6.3.5 Internal Validation Metrics

Metrics were computed exclusively on non-noise points (label ≠ −1), as silhouette and related indices are undefined for the noise class:

| Metric | Value |
|---|---|
| Silhouette Score | 0.207 |
| Davies–Bouldin Index | 1.026 |
| Calinski–Harabasz Index | 2,269.3 |

DBSCAN achieves the best Silhouette Score (0.207) and the best Davies–Bouldin Index (1.026) among all methods evaluated, indicating that — among the points assigned to clusters — the resulting groups are more compact and better separated than those produced by K-means or GMM. This reflects DBSCAN's ability to form tight, locally dense clusters rather than forcing all points into convex regions.

#### 6.3.6 Cluster Structure

The algorithm identified **20 clusters** and labelled **7,929 clients (9.6%) as noise**. The cluster distribution is highly unequal:

| Cluster | Size | Share |
|---|---|---|
| −1 (noise) | 7,929 | 9.6% |
| 0 | 50,954 | 61.8% |
| 1 | 2,091 | 2.5% |
| 2 | 2,767 | 3.4% |
| 3 | 7,774 | 9.4% |
| 4 | 4,858 | 5.9% |
| 5 | 2,441 | 3.0% |
| 6–19 | 85–899 | < 1.1% each |

Cluster 0 is a dominant mass cluster containing 61.8% of all clients. The remaining 19 clusters are structurally sharper, many exhibiting near-pure single-category or single-region profiles — a direct consequence of DBSCAN's density-based separation:

| Cluster | Dominant category | Dominant region | Delivery | Margin/Revenue |
|---|---|---|---|---|
| 0 | Mixed (Toys 29%, Diapers 16%) | CENTRAL (91%) | Courier (23%) | 0.14 |
| 1 | Large goods (22%), Toys (22%) | URAL (100%) | In-store | 0.15 |
| 2 | Large goods (23%), Toys (33%) | SOUTHERN (100%) | In-store | 0.16 |
| 3 | Large goods (17%), Toys (28%) | PRIVOLZIE (100%) | In-store | 0.15 |
| 4 | Large goods (20%), Toys (33%) | SIBERIA (100%) | In-store | 0.15 |
| 5 | Toys (51%), Textiles (29%) | CENTRAL (100%) | Pick point (100%) | 0.16 |
| 6 | Pet products (96%) | CENTRAL (100%) | Mixed | 0.16 |
| 7 | Toys (80%) | NORTH (100%) | СДЭК (100%) | 0.16 |
| 8 | Toys (52%) | CENTRAL (100%) | СДЭК (100%) | 0.02 |
| 9 | Pet products (99%) | PRIVOLZIE (100%) | In-store | 0.29 |
| 10 | Feeding products (92%) | NORTH (100%) | In-store | 0.38 |
| 11 | Large goods (100%) | NORTH (100%) | Self-pickup (100%) | 0.16 |
| 12 | Toys (100%) | NORTH (100%) | Self-pickup (100%) | 0.26 |
| 15 | Stationery/Books (96%) | PRIVOLZIE (100%) | — | 0.26 |
| 16 | Stationery/Books (95%) | NORTH (100%) | — | 0.18 |
| 17 | Hygiene (92%) | SOUTHERN (100%) | — | 0.16 |
| 18 | Home appliances (100%) | CENTRAL (100%) | Courier (6%) | 0.16 |

A key structural observation is that DBSCAN has decomposed the feature space primarily along **geographic × category × delivery method** axes, producing micro-segments that are internally very homogeneous but collectively represent a small fraction of the client base. This level of granularity has high potential for targeted marketing but requires careful interpretation to avoid acting on segments too small for statistically reliable inference.

The **noise group** (−1) contains 7,929 clients characterised by higher revenue (median 3,149), more diversified category portfolios, higher cancel ratios (22%), and mixed delivery and regional profiles — behavioural signatures that place these clients in sparse, low-density regions of the feature space without clear neighbours.

#### 6.3.7 Visualisations

**UMAP projection.** UMAP (Uniform Manifold Approximation and Projection) was applied to the full 36-dimensional scaled matrix (not the PCA reduction) with `n_neighbors=30`, `min_dist=0.1`, `n_components=2`, producing a 2D embedding. The DBSCAN cluster labels were overlaid on this projection, with noise points rendered in light grey. The UMAP plot reveals a clearly structured manifold with a large central mass (Cluster 0) surrounded by distinct peripheral islands corresponding to the specialised micro-segments — a strong visual confirmation that the geographic-category-delivery combinations identified by DBSCAN correspond to genuinely separated regions of the behavioural space.

**Pie chart.** The distribution of clients across clusters (including noise) was visualised as a pie chart, highlighting the dominance of Cluster 0 and the noise class.

---

### 7.4 HDBSCAN (UMAP + HDBSCAN 2)

#### 6.4.1 Objective

HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) extends DBSCAN by building a full cluster hierarchy across all density levels and then extracting a flat partition at the level that maximises cluster persistence. Unlike DBSCAN, which requires a fixed global neighbourhood radius ε, HDBSCAN adapts to local density variations, making it capable of finding clusters of varying density within the same dataset. It retains the noise-labelling property of DBSCAN while producing more stable and interpretable results across a wider range of parameter choices.

In this study, a two-stage pipeline was applied: the 36-dimensional scaled feature matrix was first reduced with UMAP to a 10-dimensional embedding optimised for clustering (not visualisation), and HDBSCAN was subsequently fitted on this compact representation.

#### 6.4.2 UMAP Dimensionality Reduction for Clustering

A dedicated UMAP reduction was constructed specifically for the HDBSCAN input, distinct from the 2D UMAP used for visualisation. The parameters were chosen to favour cluster preservation over spatial spread:

- `n_components = 10` — retains substantially more structure than the 2D visualisation embedding
- `n_neighbors = 30` — balances local and global structure
- `min_dist = 0.0` — collapses intra-cluster points as tightly as possible, amplifying density contrast between clusters and sparse regions
- Input: 25-component PCA representation (90% explained variance), carried over from the DBSCAN preprocessing pipeline

The resulting embedding has shape **(82,399 × 10)**.

#### 6.4.3 Grid Search over HDBSCAN Hyperparameters

HDBSCAN has two primary hyperparameters:

- **`min_cluster_size`:** the minimum number of points for a group to be considered a cluster; smaller values yield more, finer clusters.
- **`min_samples`:** controls noise sensitivity; higher values make the algorithm more conservative and label more points as noise.
- **`cluster_selection_method`:** `'eom'` (Excess of Mass, default) tends to produce fewer, larger clusters; `'leaf'` extracts the finest-grained clusters from the tree.

An 18-configuration grid search was run over `min_cluster_size` ∈ {500, 1000, 2000}, `min_samples` ∈ {5, 10, 20}, and `cluster_selection_method` ∈ {`'eom'`, `'leaf'`}. Results were sorted by noise ratio and number of clusters:

| min_cluster_size | min_samples | method | Clusters | Noise ratio |
|---|---|---|---|---|
| 500 | 20 | eom | 38 | 5.8% |
| 500 | 5 | eom | 41 | 8.3% |
| **1000** | **10** | **eom** | **24** | **8.5%** |
| 1000 | 20 | eom | 25 | 9.0% |
| 1000 | 5 | eom | 24 | 10.1% |
| 500 | 10 | eom | 45 | 11.2% |
| 1000 | 10 | leaf | 28 | 11.8% |
| 2000 | 5–20 | eom | 15–16 | 18.6–19.8% |

The configuration `min_cluster_size=1000, min_samples=10, cluster_selection_method='eom'` was selected as the optimal balance: 24 clusters, 8.5% noise — low enough to preserve coverage, high enough to ensure all clusters are statistically substantive (≥ 1,000 clients each).

#### 6.4.4 Final Model

The final HDBSCAN 2 model was fitted with:

- `min_cluster_size = 1000`
- `min_samples = 10`
- `metric = 'euclidean'`
- `cluster_selection_method = 'eom'`
- `prediction_data = True` (required for soft membership computation)

#### 6.4.5 Internal Validation Metrics

Metrics were computed on the 10-dimensional UMAP embedding, excluding noise points:

| Metric | Value |
|---|---|
| Silhouette Score | **0.620** |
| Davies–Bouldin Index | **0.599** |
| Calinski–Harabasz Index | **69,726.2** |

HDBSCAN 2 achieves by far the strongest internal validation scores of all methods tested. The Silhouette Score of 0.620 is approximately three times higher than K-means (0.148) and seven times higher than GMM (0.087). The Davies–Bouldin Index of 0.599 is well below 1 — a threshold conventionally considered indicative of good separation. The Calinski–Harabasz score of 69,726 exceeds the next best result (K-means: 3,767) by an order of magnitude.

These exceptional scores reflect two complementary factors: (1) UMAP's ability to construct a low-dimensional embedding that concentrates intra-cluster variance and maximises inter-cluster separation prior to HDBSCAN, and (2) HDBSCAN's density-adaptive cluster extraction, which naturally produces compact, well-separated groups even in non-Euclidean data geometries. Importantly, metrics are computed in the 10D UMAP space — not the original 36D space — so they are not directly comparable on an absolute basis with the other methods, which were evaluated in the original or PCA-reduced space.

#### 6.4.6 Cluster Distribution and Soft Membership

The algorithm produced **24 clusters** and labelled **6,975 clients (8.5%) as noise**. Unlike DBSCAN, the distribution across clusters is substantially more balanced — no single cluster dominates the partition:

| Cluster | Size | Cluster | Size |
|---|---|---|---|
| −1 (noise) | 6,975 | 12 | 1,602 |
| 0 | 3,646 | 13 | 2,612 |
| 1 | 2,908 | 14 | 3,408 |
| 2 | 1,895 | 15 | 1,996 |
| 3 | 3,313 | 16 | 1,909 |
| 4 | 5,796 | 17 | 2,498 |
| 5 | 1,664 | 18 | 2,635 |
| 6 | 1,198 | 19 | 5,453 |
| 7 | 4,263 | 20 | 5,111 |
| 8 | 8,112 | 21 | 2,166 |
| 9 | 1,151 | 22 | 3,125 |
| 10 | 5,729 | 23 | 1,831 |
| 11 | 1,403 | | |

The largest cluster (8) contains 8,112 clients (9.8%), the smallest (9) contains 1,151 (1.4%). This relative balance — compared to DBSCAN's Cluster 0 with 62% — indicates that the UMAP + HDBSCAN pipeline has successfully broken the feature space into meaningfully distinct, similarly-sized segments.

**Soft membership (assignment certainty).** HDBSCAN provides a membership probability for each point reflecting its confidence of cluster assignment. The distribution of maximum membership probabilities across all 82,399 clients shows:

| Statistic | Value |
|---|---|
| Mean | 0.574 |
| Median | 0.529 |
| 25th percentile | 0.190 |
| 75th percentile | 1.000 |
| Min / Max | 0.000 / 1.000 |

The bimodal character of this distribution — with a mass of clients at high certainty (≥ 1.0) and a tail of low-certainty clients — reflects the hierarchical nature of the algorithm: core points deep within dense clusters receive certainty = 1.0, while boundary and near-noise points receive intermediate values. A histogram of membership probabilities was generated to confirm this structure.

#### 6.4.7 Visualisations

**UMAP 2D projection.** The 2D UMAP embedding computed in the DBSCAN section was reused for visualisation, with HDBSCAN 2 labels overlaid. Noise points are shown in light grey; cluster points are coloured using a 20-colour palette. Compared to the DBSCAN visualisation on the same 2D projection, the HDBSCAN 2 solution covers the manifold more uniformly, with no single dominant colour region — consistent with the more balanced cluster sizes.

**Membership probability histogram.** A histogram of maximum membership probabilities across all clients confirms the bimodal distribution, with a large mass at certainty = 1.0 and a secondary peak near 0.

**Pie chart.** The distribution of cluster sizes (including noise) was visualised as a pie chart, confirming the substantially more balanced partition compared to DBSCAN.

---

### 7.5 Hierarchical Clustering (Ward Linkage)

#### 6.5.1 Objective

Agglomerative hierarchical clustering builds a nested sequence of partitions by successively merging the two closest clusters at each step, forming a binary tree (dendrogram) that encodes the complete merge history of the dataset. The Ward linkage criterion minimises the total within-cluster variance at each merge — an objective directly analogous to K-means inertia — making it one of the most geometrically principled linkage methods. The key advantage of hierarchical clustering is that it does not require the number of clusters to be fixed in advance: any flat partition can be extracted from the dendrogram by cutting it at a chosen height, and the dendrogram itself reveals the hierarchical organisation of the data at multiple granularity levels simultaneously.

#### 6.5.2 Sampling Strategy

A critical practical constraint of agglomerative clustering is its computational complexity: the standard algorithm scales as O(n² log n) in time and O(n²) in memory, making direct application to 82,399 clients infeasible. Two sampling approaches were employed:

- **Dendrogram construction and cut-point selection:** a random sample of **10,000 clients** was drawn (with `random_state=42`) and used to fit the Ward linkage matrix via `scipy.cluster.hierarchy.linkage`. This sample is large enough to faithfully reflect the hierarchical structure of the full dataset while remaining computationally tractable.
- **Metric evaluation:** a larger random sample of **20,000 clients** was used to fit `AgglomerativeClustering` and evaluate internal validation metrics, providing a more representative estimate of clustering quality.

The full dataset shape at this stage was **(82,399 × 36)** for the 36-feature standardised matrix, and **(74,435 × 30)** for an earlier intermediate version — the section was executed on the 36-feature matrix.

#### 6.5.3 Selecting the Number of Clusters

**Dendrogram inspection.** The dendrogram was plotted in truncated form (`truncate_mode='lastp'`, showing the last 40 merged super-clusters), allowing the large-scale merge structure to be visualised without rendering 10,000 individual leaves. The vertical axis (merge height under the Ward criterion) indicates the within-cluster variance added at each merge step — large jumps signal that two substantially dissimilar groups are being forced together and indicate natural cut points.

**Merge height and gap analysis.** The sequence of merge heights for the last 200 steps was plotted alongside the first-differences (gaps) between consecutive merge heights. A pronounced gap in merge heights indicates a level in the hierarchy where cluster structure becomes unstable — the optimal cut is placed just below this gap, preserving all the natural cluster separations while avoiding forced merges of genuinely distinct groups.

**Cut threshold.** A horizontal cut at merge height **t = 115** was applied to the dendrogram, yielding **19 clusters** from the 10,000-point sample. This value was selected by visual inspection of the dendrogram and the gap plot as the level below which the hierarchy is internally stable and above which the merges become large and information-destroying.

#### 6.5.4 Full-Dataset Fitting

The final model was fitted using `AgglomerativeClustering` with:

- `n_clusters = 19`
- `linkage = 'ward'`
- Applied to a 20,000-point random subsample of the full scaled matrix for metric evaluation

#### 6.5.5 Internal Validation Metrics

| Metric | Value |
|---|---|
| Silhouette Score | 0.174 |
| Davies–Bouldin Index | 1.441 |
| Calinski–Harabasz Index | 1,042.0 |

The hierarchical clustering achieves a Silhouette Score of 0.174 and a DBI of 1.441, placing it between K-means (Silhouette 0.148, DBI 1.982) and DBSCAN (Silhouette 0.207, DBI 1.026) in terms of geometric cluster quality. The lower Calinski–Harabasz score (1,042 vs. K-means's 3,767) partly reflects the larger number of clusters (19 vs. 10), since CH tends to favour solutions with fewer, denser clusters. The DBI of 1.441 is notably better than K-means, suggesting that Ward linkage produces more separated, less overlapping clusters despite operating on the same feature space.

#### 6.5.6 Comparison with Other Methods

Among the partition-based methods evaluated in the original high-dimensional feature space (K-means, GMM, Hierarchical), Ward linkage achieves the best Davies–Bouldin Index (1.441 vs. K-means 1.982, GMM 2.960), indicating superior inter-cluster separation relative to intra-cluster scatter. This is consistent with the theoretical properties of Ward linkage, which directly optimises a variance-minimisation criterion aligned with these geometric metrics.

#### 6.5.7 Visualisations

**Truncated dendrogram.** The dendrogram was plotted showing the last 40 collapsed super-clusters. Each leaf in the truncated representation corresponds to a set of merged points; leaf labels denote cluster IDs. The vertical merge height at which the horizontal cut at t = 115 intersects the tree determines the 19-cluster partition.

**Dendrogram with cut line.** A second dendrogram plot included a horizontal line at t = 115, visually marking the chosen cut point and illustrating the resulting 19 natural groupings.

**Merge height and gap curves.** Two line plots of the final 200 merge steps — one showing cumulative merge heights, one showing first-differences (gaps) — were used to identify the optimal cut point and confirm the stability of the t = 115 threshold.

---

## 8. RFM Segmentation

### 8.1 Objective

The objective of this stage was to assign each client–quarter observation to a meaningful behavioural segment based on the three classical RFM dimensions — Recency, Frequency, and Monetary value. The resulting segmentation served a dual purpose: it provided an interpretable customer taxonomy for business analysis, and it constituted the target variable for all subsequent predictive models of segment transitions and downgrade risk.

### 8.2 Methods

**RFM dimensions and scoring.** Each of the three RFM dimensions was translated into an ordinal score on a 1–5 scale using distinct binning strategies appropriate to the statistical distribution of each variable:

- **Recency (R):** Measured as `recency_days` — the number of days since the client's last purchase as of the quarter-end date. Lower values indicate more recent activity and are therefore associated with higher scores. Quintile-based breakpoints (at the 20th, 40th, 60th, and 80th percentiles) were used to convert raw recency values into scores from 1 (least recent) to 5 (most recent).

- **Frequency (F):** Measured as `total_orders` — the total number of unique orders placed by the client within the quarter. Due to the severely right-skewed and discrete distribution of this variable (over 50% of clients placed a single order per quarter, rendering quantile-based bins degenerate), frequency scoring was implemented via manually defined thresholds: 1 order → score 1; 2 orders → score 2; 3–4 orders → score 3; 5–8 orders → score 4; 9 or more orders → score 5. This approach ensured that all five score levels were reachable and meaningfully populated.

- **Monetary (M):** Measured as `margin` — the total gross margin generated by the client within the quarter. Higher values reflect greater commercial value and correspond to higher scores. Quintile-based breakpoints were used, analogous to the recency dimension, yielding scores from 1 (lowest margin) to 5 (highest margin).

**Prevention of data leakage.** Quantile breakpoints for the R and M dimensions were estimated exclusively on observations from quarters Q1 through Q3 of 2017, deliberately excluding Q4. This ensured that threshold computation was not contaminated by data from the test period, preserving the integrity of the temporal evaluation scheme.

**Segment assignment rules.** Based on the three component scores, each client–quarter was assigned to one of six segments using an ordered rule-based classifier. The priority of conditions was defined from most specific to most general, preventing overlap between segments:

| Segment | Condition | Interpretation |
|---|---|---|
| **Champions** | R ≥ 4 AND F ≥ 3 AND M ≥ 3 | Recent, frequent, and high-value customers |
| **Loyal** | R ≥ 3 AND F ≥ 3 | Regular buyers with sustained purchase activity |
| **Promising** | R ≥ 4 AND F ≤ 2 | Very recent but low-frequency — new or reactivated clients |
| **At Risk** | R ≤ 2 AND (F ≥ 3 OR M ≥ 3) | Lapsed customers who were historically active or valuable |
| **Lost** | R = 1 AND F ≤ 2 AND M ≤ 2 | Inactive across all three dimensions |
| **Regular** | All other combinations | Moderate activity, no distinguishing extreme |

Segments were additionally assigned an ordinal rank (Champions = 5, Loyal = 4, Promising = 3, Regular = 2, At Risk = 1, Lost = 0) to enable the definition of a downgrade event as a decrease in rank between consecutive quarters.

**Transition panel construction.** To support modelling of segment dynamics, the segmented dataset was restructured into a longitudinal panel format. For each client, the segment label of the immediately subsequent quarter was appended as a target variable (`segment_next`) via a one-period lag operation. This yielded a set of observed segment transitions across three consecutive quarter-pairs: Q1→Q2, Q2→Q3, and Q3→Q4.

**Temporal train/test split.** The transition panel was partitioned into a training set (transitions Q1→Q2 and Q2→Q3) and a test set (transitions Q3→Q4). This strictly chronological split prevented any form of temporal leakage, ensuring that model evaluation reflected genuine out-of-sample predictive performance.

### 8.3 Results and Conclusions

The RFM segmentation produced a six-tier customer taxonomy applied consistently across all four quarters of the observation period. The frequency scoring design choice — using manually defined thresholds rather than quantile binning — was critical for ensuring a non-degenerate score distribution, given that the majority of clients are low-frequency buyers. The rule-based segment assignment scheme, with its clearly ordered priority conditions, guaranteed mutually exclusive and collectively exhaustive segment membership. The resulting longitudinal panel, covering approximately 15,000 unique clients across four quarters, formed the foundation for all subsequent modelling of segment stability, downgrade prediction, and survival analysis.

---

## 9. Comparative Summary of Clustering Methods

### 9.1 Internal Validation Metrics

The table below consolidates the internal validation metrics for all five clustering methods. Metrics for DBSCAN and hierarchical clustering were computed on non-noise points only; HDBSCAN 2 metrics were computed in the 10-dimensional UMAP space and are therefore not directly comparable to the others on an absolute scale.

| Method | k / clusters | Noise % | Silhouette ↑ | DBI ↓ | CH ↑ |
|---|---|---|---|---|---|
| K-means | 10 | 0% | 0.148 | 1.982 | 3,767 |
| GMM (EM) | 11 | 0% | 0.087 | 2.960 | 2,364 |
| DBSCAN | 20 (+noise) | 9.6% | 0.207 | **1.026** | 2,269 |
| HDBSCAN 2 (UMAP) | 24 (+noise) | 8.5% | **0.620*** | **0.599*** | **69,726*** |
| Hierarchical (Ward) | 19 | 0% | 0.174 | 1.441 | 1,042 |

*\* Computed in 10D UMAP space — not directly comparable to other methods.*

### 9.2 Qualitative Comparison

| Criterion | K-means | GMM | DBSCAN | HDBSCAN 2 | Hierarchical |
|---|---|---|---|---|---|
| Requires k in advance | Yes | Yes | No | No | No (dendrogram) |
| Handles noise/outliers | No | No | Yes | Yes | No |
| Cluster shape assumption | Spherical | Ellipsoidal | Arbitrary | Arbitrary | Arbitrary |
| Soft assignments | No | Yes | No | Yes | No |
| Scalable to 82K clients | Yes | Slow | Yes (via PCA) | Yes (via UMAP) | Requires sampling |
| Segment balance | Uneven | Moderate | Very uneven | Balanced | Moderate |
| Interpretability | High | Moderate | Low (micro-segments) | Moderate | High (dendrogram) |

### 9.3 Conclusions

Each method reveals a distinct facet of the client population's structure:

- **K-means** provides a fast, interpretable 10-cluster partition with clear financial differentiation (Cluster 9 as premium, Clusters 4/8 as low-margin), suitable as a baseline for business reporting.
- **GMM** uncovers more nuanced behavioural segments by modelling ellipsoidal cluster shapes, identifying near-pure single-category groups (Cluster 9: 99% toys; Cluster 2: 61% footwear) and the only entirely cash-paying courier segment (Cluster 3) that K-means merges with others.
- **DBSCAN** produces the sharpest geometric separation among the partition-based methods (DBI 1.026) and reveals that the feature space decomposes primarily along geographic × category × delivery axes, yielding 20 highly homogeneous micro-segments alongside a dominant mass cluster of 62%.
- **HDBSCAN 2** achieves the most balanced partition (24 clusters, 1,151–8,112 clients each) and the strongest internal validation scores by a wide margin in its UMAP embedding space, making it the most structurally coherent solution at the cost of reduced direct interpretability.
- **Hierarchical clustering** provides the only method that explicitly exposes the multi-level structure of the data through a dendrogram, allowing the analyst to read off partitions at any desired granularity without refitting — a valuable property for exploratory analysis and stakeholder communication.
- **RFM segmentation** complements the unsupervised methods with an immediately actionable, business-interpretable layer: it identifies that 5.9% of clients (Big Spenders) generate 16.4% of revenue, that 14.8% of revenue is at risk due to high-value client churn (At Risk segment), and that 15% of clients are newly acquired single-purchase buyers with high conversion potential.
