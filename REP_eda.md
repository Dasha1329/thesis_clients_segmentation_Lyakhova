# Exploratory Data Analysis Report
## Customer Segmentation Based on Transactional Data

---

> **Source notebook:** `eda.ipynb`  
> **Dataset:** `transactions_2017.parquet`  
> **Author:** Daria  
> **Date:** April 2026

---

## Table of Contents

1. [Data Loading and Preprocessing](#1-data-loading-and-preprocessing)
2. [Missing Value Imputation and Geographic Enrichment](#2-missing-value-imputation-and-geographic-enrichment)
3. [Exploratory Data Analysis of Transactional Data](#3-exploratory-data-analysis-of-transactional-data)

---

## 1. Data Loading and Preprocessing

### 1.1 Objective

The primary objective of this stage was to load the raw transactional dataset, assess its structure and volume, identify data quality issues related to incorrect data types and malformed field values, and perform the necessary corrections to produce a clean, analysis-ready DataFrame. This step is foundational: all subsequent exploratory analysis and RFM-based segmentation depend on the integrity of the data established here.

---

### 1.2 Dataset Overview

The dataset was loaded from a Parquet file (`transactions_2017.parquet`) containing order-level records for the full calendar year 2017. Upon loading, the DataFrame comprised **4,122,221 rows and 40 columns**, where each row corresponds to a single line item within a customer order. The key fields present in the dataset include:

| Field | Description |
|---|---|
| `Дата` | Order placement date and time |
| `ДатаДоставки` | Delivery date |
| `НомерЗаказаНаСайте` | Website order number (unique order identifier) |
| `Статус` / `НовыйСтатус` | Order and delivery status |
| `СуммаЗаказаНаСайте` | Total order value as displayed on the website |
| `СуммаДокумента` | Document (invoice) amount |
| `Цена`, `СуммаСтроки` | Unit price and line item total |
| `Маржа` | Margin per line item |
| `МетодДоставки` | Delivery method (courier, pickup, store, etc.) |
| `ФормаОплаты` | Payment method (cash, card, etc.) |
| `Группа2` | Product category |
| `Регион`, `ГородМагазина` | Geographic fields |
| `Телефон_new` | Anonymised customer phone identifier |

---

### 1.3 Data Type Corrections

#### 1.3.1 Date Fields

The `Дата` column was stored as a string in day-first format and required explicit parsing using `pd.to_datetime` with `dayfirst=True`. The same conversion was subsequently applied to `ДатаДоставки` and `ДатаЗаказаНаСайте`. The `errors="coerce"` parameter ensured that any unparseable values were silently converted to `NaT` rather than raising exceptions, allowing the pipeline to continue robustly.

After conversion, the distribution of transaction volumes was verified across both date dimensions. The monthly record counts by **delivery date** (`ДатаДоставки`) were as follows:

| Month | Transaction rows |
|---|---|
| January 2017 | 205,595 |
| February 2017 | 280,810 |
| March 2017 | 291,313 |
| April 2017 | 331,846 |
| May 2017 | 293,737 |
| June 2017 | 356,324 |
| July 2017 | 290,397 |
| August 2017 | 409,531 |
| September 2017 | 303,472 |
| October 2017 | 449,998 |
| November 2017 | 400,037 |
| December 2017 | 509,161 |

The distribution by **order placement date** (`Дата`) shows a broadly similar pattern, confirming full coverage of calendar year 2017 with no missing months. Both series exhibit a pronounced growth trend from January through December, with December being the most active month — consistent with the seasonal peak in retail activity observed in the pre-holiday period.

#### 1.3.2 Numeric Fields

Multiple monetary and quantity columns were stored as strings with space-separated thousands (e.g., `"2 800"`) and comma-used-as-decimal separators — a common artefact of Russian-locale data exports. A unified cleaning procedure was applied to eight financial columns: `СуммаЗаказаНаСайте`, `СуммаДокумента`, `Цена`, `СуммаСтроки`, `ЦенаЗакупки`, `Маржа`, `СуммаУслуг`, and `СуммаДоставки`. The procedure strips non-breaking spaces (`\xa0`), regular spaces used as thousands separators, and replaces decimal commas with periods before casting to `float64`. Values that could not be parsed after cleaning were set to `NaN` for downstream imputation.

#### 1.3.3 Categorical Text Fields

The `МетодДоставки` (delivery method) and `ФормаОплаты` (payment method) columns were standardised to lowercase and stripped of leading/trailing whitespace to eliminate duplicate categories caused by inconsistent casing.

---

### 1.4 Customer Identifier Cleaning (`Телефон_new`)

The `Телефон_new` column serves as the primary customer identifier throughout the analysis. It contains anonymised phone numbers in hashed form. An audit of the value formats revealed four distinct patterns:

| Format type | Count | Share |
|---|---|---|
| Dash-separated hash (e.g., `55575454-49504949555170`) | 3,202,743 | 77.7% |
| Digits only (including `0` placeholders) | 16,371 | 0.4% |
| Date-like strings (e.g., `21.11.2017 9:59`) | 1,326 | 0.03% |
| Other (mixed characters, `None` strings) | 901,781 | 21.9% |

The **date-like entries** represent a data quality anomaly: a timestamp value appears to have been erroneously written into the phone identifier field. These 1,326 rows were identified using a regular expression and removed from the dataset. After removal, the mask was re-applied and confirmed to return zero matches, validating the cleaning step.

---

### 1.5 Monthly Revenue Dynamics

As an initial view of business activity, monthly revenue was aggregated across all orders using the order placement date. The resulting bar chart (revenue in millions of roubles, January–December 2017) reveals a clear upward trend throughout the year. Revenue grows steadily from January through mid-year, with a notable acceleration in the autumn months (September–November), and reaches its peak in December. This seasonal pattern is typical for e-commerce in the children's goods segment, where demand intensifies around the New Year holiday period.

The chart serves as a high-level sanity check confirming that (a) all twelve months are represented, (b) revenue values are positive and plausible after numeric cleaning, and (c) the business demonstrated consistent growth over the observation period — an important contextual factor for interpreting the subsequent RFM segmentation.

---

### 1.6 Summary

By the end of this preprocessing stage, the dataset had been cleaned as follows:

- Date columns parsed to `datetime64` with `dayfirst=True`
- Eight numeric columns converted from locale-formatted strings to `float64`
- Categorical delivery and payment fields normalised to lowercase
- 1,326 rows with corrupted phone identifier values removed
- Final dataset retained for downstream analysis: **~4,120,895 rows**

The data covers 12 consecutive months of 2017 with no missing months, and the initial revenue trend confirms the business was growing throughout the year.

---

## 2. Missing Value Imputation and Geographic Enrichment

### 2.1 Objective

Following the initial type corrections, this stage addressed the remaining data quality issues: incomplete customer identifiers, structurally invalid rows, missing geographic values, and the absence of a standardised macro-regional classification. The goal was to produce a fully resolved, analytically consistent dataset in which every retained row carries a valid customer identifier and an unambiguous geographic assignment at the macro-region level.

---

### 2.2 Customer Identifier Audit and Final Cleaning

#### 2.2.1 Type Composition of `Телефон_new`

A systematic audit of the customer identifier column revealed that the field contained a mixture of Python types at the object level:

| Python type | Row count |
|---|---|
| `str` | 3,219,142 |
| `NoneType` | 901,753 |

Explicit diagnosis of the string values then uncovered three categories of invalid entries. The 901,753 `NoneType` values serialised to the string `"None"` upon string conversion — a common Pandas behaviour that must be handled explicitly. The 16,371 rows carrying the value `"0"` represent a placeholder used in source systems when the customer's phone number was unavailable. All three categories — true nulls, `"0"` placeholders, and `"None"` strings — were removed sequentially, reducing the dataset to rows with valid, non-trivial customer identifiers.

#### 2.2.2 Removal of Delivery Service Line Items

Rows in which the `Номенклатура` (product name) or `Группа2` (product category) field equalled `"Доставка"` (delivery) represent shipping fee line items rather than product purchases. These were removed to ensure that product-level financial aggregations are not distorted by non-product revenue.

#### 2.2.3 Column Rationalisation

After cleaning, the dataset was reduced from 40 raw fields to a curated set of 26 analytically relevant columns, logically grouped as: customer identifier, order metadata, logistics and payment, product hierarchy, quantity fields, and financial fields. This reduces memory footprint and prevents accidental use of auxiliary source-system columns in downstream analysis.

**Dataset shape after this stage: 2,732,793 rows × 26 columns.**

---

### 2.3 Missing Value Analysis

A full null count was produced for all 26 retained columns:

| Field | Null count | Nature |
|---|---|---|
| `Регион` | 18,304 | Missing geographic assignment — requires imputation |
| `ПричинаОтмены` | 2,430,338 | Structural: only cancelled orders have a cancellation reason |
| `ЦенаЗакупки` | 10,775 | Missing purchase price — rows later filtered via `Маржа` |
| All other fields | 0 | Complete |

The `ПричинаОтмены` nulls are not a data quality problem: the vast majority of orders are not cancelled, so the field is legitimately empty. It was retained without imputation. The `ЦенаЗакупки` nulls propagate into `Маржа` and are resolved at a later filtering step. The primary imputation target was the `Регион` field.

---

### 2.4 Geographic Field Cleaning and City Name Normalisation

#### 2.4.1 Bracket Removal

The raw `Регион` field stored city names together with district annotations in parentheses (e.g., `"Люберцы (Люберецкий район)"`, `"Жуковский (Московская область район)"`). These were stripped using a regular expression, yielding clean city names.

#### 2.4.2 Fallback Imputation from `ГородМагазина`

Where `Регион` remained null after bracket removal but a non-null store city (`ГородМагазина`) was available, the store city was used as a proxy for the customer's geographic location. The column was subsequently renamed from `Регион` to `Город` to reflect that it now holds a city name rather than an administrative region, pending the forthcoming macro-region join.

#### 2.4.3 String Normalisation for Joining

To enable a reliable string join with the external reference table, city names in both the transaction data and the reference file were normalised by: removing residual parenthetical suffixes, converting to lowercase, stripping leading/trailing whitespace, and replacing `"ё"` with `"е"` — the two letters are treated interchangeably in most Russian geographic databases.

#### 2.4.4 Manual Pre-Mapping of Moscow Suburbs

A set of Moscow-area settlements that are administratively distinct but commercially part of the Moscow metropolitan market were manually remapped to `"москва"` prior to joining. This ensures that customers from satellite towns (e.g., Путилково, Щербинка, Зеленоград, Коммунарка, Внуково) are correctly attributed to the CENTRAL macro-region rather than left unmatched.

---

### 2.5 Macro-Region Lookup Join

An external reference table (`data/для_сас_города+_население+мелкие_города.xlsx`, sheet *"население 2014"*) containing 656 Russian cities, their population (2014 census), and macro-region assignments was used to enrich the transaction data with a standardised regional classification. The macro-regions defined in the reference are: **CENTRAL**, **NORTH**, **PRIVOLZIE**, **SIBERIA**, **FAR EAST**, **SOUTHERN**, **URAL**.

The join was performed as a left merge on the normalised city name. After the initial merge, **27,146 rows** remained unmatched (i.e., the city name was not found in the reference table). Inspection of the top unmatched values revealed that most were peri-urban settlements in the Moscow region or small towns absent from the reference:

| City | Unmatched rows |
|---|---|
| котельники | 3,946 |
| крым | 3,252 |
| нахабино | 1,371 |
| апрелевка | 1,104 |
| малаховка | 940 |
| звенигород | 670 |
| лосино-петровский | 606 |

A secondary manual mapping dictionary of approximately 50 additional settlements was constructed and applied to resolve the majority of these cases. After this second pass, **2,856 rows** remained unresolved — primarily rows with ambiguous partial city names (e.g., `"сергиев"`, `"старый"`, `"нижний"` without a disambiguating second word) or null city values. These rows were dropped.

---

### 2.6 Regional Summary Statistics

Following successful region assignment, aggregate statistics were computed at the macro-region level to validate geographic coverage and characterise regional customer bases:

| Macro-region | Unique customers | Revenue (RUB) | Margin (RUB) | Unique orders | Avg. order value (RUB) |
|---|---|---|---|---|---|
| CENTRAL | 235,665 | 1,686,689,000 | 379,112,100 | 477,463 | 3,533 |
| PRIVOLZIE | 50,786 | 312,518,800 | 66,256,680 | — | — |
| NORTH | 36,138 | 177,462,800 | 36,603,170 | — | — |
| SOUTHERN | 21,660 | 155,365,200 | 34,501,270 | — | — |
| URAL | 20,603 | 134,631,400 | 29,879,730 | — | — |
| SIBERIA | 19,592 | 124,040,300 | 27,209,100 | — | — |
| FAR EAST | 1,228 | 10,219,510 | 1,752,790 | 3,731 | 2,739 |

The CENTRAL macro-region (Moscow and surrounding areas) accounts for the dominant share of customers, revenue, and order volume — a structural feature of the Russian e-commerce market that must be accounted for when constructing and interpreting customer segments.

---

### 2.7 Final Column Selection and Margin Filtering

The dataset was reduced to 23 analytically relevant columns, excluding auxiliary source-system fields that are not required for segmentation. Rows with a null `Маржа` value were then dropped, as margin is a required field for the subsequent RFM monetary component. After this filter, a null check confirmed that all 21 analytical fields (excluding the structurally sparse `ПричинаОтмены`) were complete, with zero missing values.

---

### 2.8 Temporal Coverage Check and Quarter Feature

A final distribution of transaction counts by month was computed to verify temporal integrity:

| Month | Row count |
|---|---|
| Jan–Oct 2017 | 257,694 – 333,801 (per month) |
| November 2017 | **2,829** |
| December 2017 | **2,238** |

The sharp drop in November and December indicates that the cleaned dataset provides full coverage only through October 2017. November and December contain marginal data and are treated as incomplete months in subsequent analyses. This is an important boundary condition for the RFM model: the snapshot date and the effective observation window should be understood as reflecting January–October 2017 behaviour. A quarter indicator was added as a convenience feature for period-level aggregations.

**Final dataset shape: 2,719,182 rows × 22 columns.**

---

## 3. Exploratory Data Analysis of Transactional Data

### 3.1 Objective

The purpose of this stage was to characterise the transactional dataset along its main business dimensions — order fulfilment quality, revenue and margin structure, customer purchasing behaviour, and operational patterns — prior to constructing the RFM model. The analysis moves from aggregate order-level statistics to customer-level distributions, establishing the empirical basis for subsequent segmentation.

---

### 3.2 Order Status Distribution and Return Dynamics

#### 3.2.1 Order Status Composition

Order statuses were aggregated at the level of unique orders. The pie chart of status shares reveals the dominant outcome categories: the majority of orders are recorded as delivered (`Доставлен`), with meaningful proportions attributed to cancellations (`Отменен`) and returns (`Возврат`). A smaller share falls under transitional statuses such as *"К отгрузке"* (awaiting dispatch) and *"Возврат из ПВЗ"* (pickup-point return).

This decomposition is critical for downstream analysis: only orders with status `Доставлен` are used to compute RFM metrics, ensuring that cancelled and returned transactions do not inflate customer value or recency estimates.

#### 3.2.2 Monthly Return Rate

The share of returned orders relative to total orders placed was computed month-by-month and visualised as a line chart. The resulting series shows modest month-to-month variation in the return rate across 2017, without a pronounced seasonal spike. This stability suggests that the return process is not strongly linked to seasonal demand fluctuations and reflects a structurally consistent fulfilment quality throughout the year.

---

### 3.3 Order Value and Basket Composition

#### 3.3.1 Distribution of Order Values

Orders were collapsed to order-level records, with the order value taken as the first recorded `СуммаЗаказаНаСайте` per order identifier. The distribution was visualised up to the 95th percentile to reduce the visual impact of outliers:

| Statistic | Value |
|---|---|
| Median order value | **2,199 RUB** |
| Mean order value | **3,494 RUB** |

The distribution is strongly right-skewed: the bulk of orders are clustered in the 1,000–5,000 RUB range, while the mean is pulled upward by a long tail of high-value orders. The median is therefore a more representative measure of the typical transaction. The vertical median line overlaid on the histogram makes this asymmetry visually explicit.

#### 3.3.2 Number of SKUs per Order

The number of line items (distinct positions) per order was computed as the row count per order identifier. The bar chart of SKU counts (capped at 15 items) demonstrates that most orders consist of a small number of items — typically one to three positions — consistent with focused, low-basket-depth purchasing behaviour characteristic of e-commerce in the children's goods segment.

---

### 3.4 Revenue and Margin Structure

#### 3.4.1 By Product Category

Revenue and margin were summed across all transactions by product category and displayed as horizontal bar charts ranked by value. The revenue ranking identifies the highest-turnover product groups, which are dominated by categories with high purchase frequency (e.g., nappies, textiles). The margin chart reveals a more differentiated picture: some high-revenue categories carry comparatively lower absolute margin, reflecting different pricing and cost structures. Categories such as large-format goods tend to generate higher per-order revenue but may show lower margin density. For the purposes of segmentation, the category-level margin structure is important context: customers concentrated in high-margin categories represent disproportionate commercial value relative to their revenue contribution alone.

#### 3.4.2 By Region

Geographic revenue and margin distributions were computed analogously, aggregating by the `Регион` field. The horizontal bar charts confirm a strong geographic concentration: Moscow and St. Petersburg collectively account for the dominant share of both revenue and margin, as expected given the population size and purchasing power of these metropolitan areas. Remaining regions contribute a long tail of individually smaller but collectively significant volume.

---

### 3.5 Delivery Method and Payment Form

#### 3.5.1 Delivery Method

Unique order counts were aggregated by delivery method. The chart shows that self-pickup (`Самовывоз`) and courier delivery (`Курьерская`) together account for the majority of orders, with in-store collection (`Магазины`) representing a substantial third channel. The relative shares of these methods are relevant to operational cost analysis and to interpreting return/cancellation rates by channel.

#### 3.5.2 Payment Method

The pie chart of payment form shows a two-category split between cash (`Наличная`) and non-cash / card (`Безналичная`) payments. The relative balance between these reflects the maturity of digital payment adoption in the customer base and has implications for identifying high-value customer segments.

---

### 3.6 Customer Purchase Frequency

The distribution of customers by total number of unique orders placed was computed over the full dataset (all statuses):

| Segment | Count | Share |
|---|---|---|
| 1 order only | **247,333** | **64.2%** |
| 2–3 orders | — | 24.5% |
| 4+ orders | **43,521** | **11.3%** |

The data reveal a pronounced concentration of one-time buyers: nearly two-thirds of all identified customers placed exactly one order during the observation year. This is consistent with the typical first-purchase-heavy acquisition curve in e-commerce and underscores the commercial importance of distinguishing loyal repeat buyers (4+ orders) from occasional and one-time customers. The histogram (capped at 10 orders) and the grouped bar chart together make this structure visually unambiguous.

---

### 3.7 Temporal Activity Heatmap

Order activity was cross-tabulated by month (rows) and day of the week (columns) and displayed as an annotated heatmap coloured on a yellow-to-red scale. Two consistent patterns emerge. First, **weekdays dominate weekend days** across all months — Monday through Friday carry significantly higher order volumes than Saturday and Sunday, suggesting that the customer base places orders predominantly during working hours. Second, **order volumes grow monotonically through the year**, with autumn months (September–October) showing the highest per-day counts. These temporal patterns are relevant for operational planning and for interpreting recency values in the RFM model.

---

### 3.8 Average Order Value and Margin by Category (Delivered Orders Only)

To obtain a commercially meaningful picture, average order value and total margin were computed exclusively over delivered orders, eliminating the distortion introduced by cancellations and returns. The dual horizontal bar chart ranks categories by total margin and overlays average order value. This comparison identifies categories that are simultaneously high-value per transaction and high-margin in aggregate — these are the strategically most important product groups and should inform the interpretation of which customer segments carry the highest lifetime value.

---

### 3.9 Cancellation and Return Rates by Delivery Method

The share of delivered, returned, and cancelled orders was computed as a percentage within each delivery method and displayed as a stacked horizontal bar chart. Only delivery methods with more than 1,000 total orders were retained to ensure statistical reliability. The chart reveals that cancellation and return rates vary meaningfully across delivery channels. Courier delivery tends to exhibit a different cancellation profile compared to self-pickup, partly due to the higher friction associated with receiving a delivery at a specific time. This finding motivates including delivery method as a stratification variable when interpreting segment-level behaviour.

---

### 3.10 Preliminary RFM Component Distributions

As a final preparatory step before formal segmentation, the three RFM dimensions were computed for each customer, restricted to delivered orders. The snapshot date was set to the maximum observed order date (31 December 2017), so recency is measured as the number of days between the customer's last delivered order and the end of the observation window. The resulting RFM table covers **294,783 unique customers**.

| Metric | Mean | Median (50%) | Q1 (25%) | Q3 (75%) | Max |
|---|---|---|---|---|---|
| Recency (days) | 191.9 | 181.0 | 107.0 | 269.0 | 363 |
| Frequency (orders) | 1.8 | 1.0 | 1.0 | 2.0 | 932 |
| Monetary (RUB) | 31,683 | 7,345 | 3,078 | 21,210 | 13,049,632 |

The three histograms visualise the distributional shape of each component:

**Recency** is broadly distributed across the full year (29–363 days), with no dominant spike — customers are spread relatively uniformly across the recency spectrum, though a modest concentration exists around the 100–200-day range. This spread justifies the use of quantile-based scoring rather than fixed thresholds.

**Frequency** is extremely right-skewed: the median customer placed exactly one delivered order, and the vast majority placed no more than two or three. The histogram is capped at 10 orders to make this concentration visible; beyond this threshold, the distribution forms an extremely long tail reaching 932 orders for the most active customer (likely a reseller or data anomaly).

**Monetary** is similarly skewed, with a median of 7,345 RUB but a mean of 31,683 RUB driven by extreme outliers (the 95th percentile was used as the histogram cap). The Q1–Q3 interquartile range of 3,078–21,210 RUB captures the revenue band within which most customers operate.

These distributional characteristics — strong right skew in frequency and monetary, broad spread in recency — confirm that quantile-based RFM scoring (quintile or quartile assignment) is the appropriate methodology, as it is robust to outliers and does not assume any particular distributional form.

---
