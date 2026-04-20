# Customer Segment Downgrade Analysis — K-means Variant

**Notebook:** `overflows k-means.ipynb`  
**Author:** Daria  
**Date:** April 2026

> **Scope of this document.** This notebook replicates the full analytical pipeline of `overflows RFM new.ipynb` with one substitution: RFM-named segments are replaced by K-means clusters. Only the differences from the RFM pipeline are documented here; all methodology that is identical (preprocessing, survival analysis methods, business application structure) is described in `downgrade_report.md`. Note that the two notebooks use different evaluation periods: this notebook evaluates on Q3→Q4, while `overflows RFM new.ipynb` evaluates on Q2→Q3.

---

## 1. Segmentation: K-means vs RFM

### 1.1 Clustering setup

K-means was trained on Q1 2017 data using the same 35-feature matrix as the binary classifiers (heavy-tail features log-transformed and RobustScaler-scaled on Q1, then applied to all subsequent quarters). The number of clusters was set to **K = 10** (vs 6 named segments in RFM). The scaler parameters (margin shift, RobustScaler statistics) were frozen on Q1 and applied without refitting to Q2–Q4 to prevent leakage.

### 1.2 Ordinal rank assignment

RFM assigns a fixed hierarchy by segment name (Champions = 5 → Lost = 0). K-means clusters have no intrinsic order and must be ranked post-hoc. Each cluster was assigned an ordinal rank 0–9 via a **weighted composite score** computed from Q1 median/mean profiles:

| Component | Direction |
|---|---|
| `avg_order_value` | ↑ better |
| `margin` | ↑ better |
| `delivered_ratio` | ↑ better |
| `cancel_ratio` | ↓ better |

The resulting mapping (cluster_id → cluster_rank) was fixed on Q1 profiles and applied to all quarters. The cluster-id-to-rank correspondence obtained was: `{7:0, 2:1, 3:2, 9:3, 6:4, 0:5, 8:6, 1:7, 5:8, 4:9}`.

Unlike RFM's named segments, the K-means ranks carry no semantic label — rank 9 is "the best cluster by composite score" rather than "Champions". This distinction matters for the business application layer (Section 6).

### 1.3 Downgrade definition

Formally identical to RFM: `downgrade = 1` if `cluster_rank_next < cluster_rank_current`. The scale is 0–9 instead of 0–5, but the binary encoding is the same.

---

## 2. Class Distribution Shift (Critical Difference from RFM)

The substitution of K-means for RFM fundamentally changes the class distribution of the downgrade label. Note that the two pipelines now also differ in their evaluation period: the updated RFM pipeline evaluates on Q2→Q3, while K-means retains Q3→Q4 as the test set.

| Dataset | RFM downgrade rate | K-means downgrade rate |
|---|---|---|
| Train | 47.5% (Q1→Q2) | **29.6%** (Q1→Q2 + Q2→Q3) |
| Test | 46.5% (Q2→Q3) | **22.2%** (Q3→Q4) |

In the updated RFM pipeline, training and test class distributions are near-identical (47.5% vs 46.5%), with no meaningful distributional shift. In the K-means pipeline, the Q3→Q4 test downgrade rate is **22.2%** — the downgrade class is the **minority class** in the test set (3,377 vs 11,867 no-downgrade). This has the following consequences:

- K-means has a moderate distributional shift (29.6% → 22.2%), while the updated RFM pipeline has essentially none.
- The binary classification tasks differ structurally: K-means models must identify a minority downgrade class, while RFM models operate on a near-balanced distribution.
- Optimal thresholds in K-means are near 0.5–0.6 (close to the default), whereas the updated RFM pipeline uses thresholds in the 0.31–0.32 range — lower than default but much higher than the 0.17–0.20 required in the original RFM setup under distributional shift.

---

## 3. Markov and Bayesian Baseline

The K-means transition matrix operates on 10×10 states rather than 6×6. Key structural differences from RFM:

**Performance:**

| Model | Accuracy | Log-loss |
|---|---|---|
| K-means Markov/Bayesian | **53.5%** | **1.432** |
| RFM Markov/Bayesian | 22.7% | 1.702 |

The K-means Markov baseline achieves substantially higher accuracy than its RFM counterpart. This is primarily a consequence of the segmentation design rather than predictive superiority: K-means clusters are trained on the same behavioural features used for prediction, so cluster labels encode finer-grained information than the coarser RFM segment names. The higher diagonal stability of K-means clusters (clients are more likely to remain in the same cluster quarter-to-quarter) also mechanically inflates accuracy. Additionally, the two baselines are evaluated on different test periods (Q3→Q4 for K-means, Q2→Q3 for RFM) with different target definitions, so the numbers reflect different tasks and are not directly comparable.

---

## 4. Binary Classification of Downgrade

### 4.1 Feature matrix

One structural difference: instead of one-hot encoding 6 RFM segment names, the feature matrix includes `cluster_rank` as a single ordinal integer column. The total feature count is 36 (vs 35 in RFM).

### 4.2 Results at optimal threshold

All six classifiers evaluated on the Q3→Q4 test set:

| Model | Threshold | ROC-AUC | F1 | Recall | Precision |
|---|---|---|---|---|---|
| XGBoost | 0.589 | **0.883** | 0.637 | — | — |
| LightGBM | 0.630 | 0.882 | 0.639 | 0.685 | 0.599 |
| **Random Forest** | 0.600 | 0.882 | **0.641** | 0.720 | 0.578 |
| MLP | 0.604 | 0.879 | 0.629 | — | — |
| Logistic Regression | 0.604 | 0.872 | 0.620 | — | — |
| SVM | 0.399 | 0.856 | 0.610 | — | — |

**Note on comparability with RFM.** A direct metric comparison between K-means and RFM classifiers is not methodologically valid. The two pipelines differ on four compounding dimensions simultaneously: (1) the target variable (`downgrade` is defined on a 10-rank scale vs a 6-segment scale, so the two binary labels are not the same event for the same client); (2) the test period (Q3→Q4 vs Q2→Q3); (3) the training period and volume (two quarters vs one); (4) class prevalence in the test set (22.2% vs 46.5% downgrade). Because F1 depends directly on class prevalence, and ROC-AUC is influenced by the underlying data distribution and prediction difficulty, no observed metric difference can be attributed to segmentation quality alone. The K-means results above should be interpreted on their own terms, not as a ranking against the RFM pipeline.

**Model selection.** **LightGBM** was selected as the primary model for the downstream business application, as it achieved the highest or near-highest AUC across trials and had numerically stable feature importances for SHAP analysis.

**Threshold interpretation.** Optimal thresholds in K-means (~0.59–0.63) are close to the conventional 0.5, reflecting near-balanced probability calibration when the downgrade class is the minority (22.2%). This contrasts structurally with the RFM pipeline, where lower thresholds (0.31–0.32) were needed to prioritise recall — itself a consequence of the different class distribution and downgrade definition.
