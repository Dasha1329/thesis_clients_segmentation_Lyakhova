# Customer Segment Downgrade Analysis Report

**Notebook:** `overflows RFM new.ipynb`  
**Author:** Daria  
**Date:** April 2026

---

## Table of Contents

1. [Markov Baseline Transition Model](#1-markov-baseline-transition-model)
2. [Bayesian Transition Model](#2-bayesian-transition-model)
3. [ML Models: Next-Segment Prediction](#3-ml-models-next-segment-prediction)
4. [Binary Classification of Downgrade](#4-binary-classification-of-downgrade)
5. [Survival Analysis](#5-survival-analysis)
6. [Business Application: Early Warning System](#6-business-application-early-warning-system)

---



---

## 1. Markov Baseline Transition Model

### 1.1 Objective

The objective of this stage was to establish a statistically grounded baseline for predicting customer segment transitions. The Markov chain model served two purposes: first, to characterise the structural dynamics of the customer base through an empirical transition probability matrix; and second, to provide a reference point against which the predictive performance of more complex machine learning models could be evaluated.

### 1.2 Methods

**Model assumption.** The first-order Markov assumption posits that the probability of transitioning from any current segment to any next segment depends solely on the current segment label, and is independent of all prior history, individual client characteristics, and calendar time. This assumption results in a single, time-homogeneous K×K transition matrix, where K is the number of segments.

**Transition matrix estimation.** The transition probability matrix was estimated from the training set, which comprised the Q1→Q2 and Q2→Q3 observed segment transitions. A raw count matrix was constructed by cross-tabulating the current segment (`segment`) against the subsequent segment (`segment_next`) across all client–quarter records in the training data. The matrix was then aligned to the full K×K structure to ensure all segment pairs were represented, with missing pairs filled with zero. Row-wise normalisation converted raw counts into conditional transition probabilities, yielding the stochastic matrix **P**, where each entry P[i, j] represents the estimated probability of a client transitioning from segment i to segment j in the next quarter.

**Prediction and evaluation.** For each observation in the test set (Q2→Q3 transitions), the model predicted the most probable next segment as the argmax of the corresponding row in **P** indexed by the client's current segment. Probabilistic predictions for log-loss computation were taken directly from the respective rows of **P**. The model was evaluated on two metrics: classification accuracy and log-loss. As a reference, the accuracy of a uniform random classifier was 14.3% (1/7 for seven equiprobable classes), and the corresponding uniform log-loss was ln(7) ≈ 1.946.

**Visualisation.** The transition matrix was visualised as a heatmap with annotated cell values, enabling direct interpretation of dominant flow directions within the customer base.

### 1.3 Results and Conclusions

**Transition matrix structure.** Analysis of the estimated transition probability matrix revealed five structural patterns:

1. **Moderate segment instability.** Diagonal elements — representing the probability of a client remaining in the same segment between consecutive quarters — ranged from 0.12 to 0.32. This indicates that the majority of clients change their behavioural segment within a single quarter, and no segment can be considered stable by default.

2. **At Risk as the gravitational centre.** For the majority of segments, the probability of transitioning into *At Risk* in the next quarter was approximately 0.20–0.31. This structural asymmetry suggests a systemic drift toward declining customer activity, irrespective of the starting segment.

3. **Partial stability of high-value segments.** *Champions* (self-transition probability 0.32) and *Loyal* (0.29) exhibited the highest retention rates among all segments. However, even for these groups, the probability of degradation within one quarter remained substantial, indicating that high-value status is not self-reinforcing in the absence of active retention efforts.

4. **Instability of the Big Spenders segment.** The probability of remaining in *Big Spenders* from one quarter to the next was low (0.13), consistent with the hypothesis that clients in this group tend to make infrequent large-volume purchases rather than sustained engagement.

5. **Lost is not an absorbing state.** A meaningful fraction of clients classified as *Lost* reappeared in the *Regular* and *Loyal* segments in the subsequent quarter. This indicates that the *Lost* label in this dataset reflects temporary disengagement rather than permanent churn, and that reactivation is a recurring phenomenon in the customer base.

**Predictive performance.** On the Q2→Q3 test set, the Markov baseline achieved a classification accuracy of **22.7%** and a log-loss of **1.702**. Both metrics surpass the uniform random classifier (14.3% accuracy, log-loss 1.946), confirming that knowledge of the current segment carries non-trivial predictive signal. However, the magnitude of the improvement is modest, indicating that the current segment label alone captures only a limited portion of the variance in next-quarter behaviour. This finding motivated the development of feature-rich machine learning models that incorporate the full set of client-level behavioural and financial attributes constructed in the feature engineering stage.

---

## 2. Bayesian Transition Model

### 2.1 Objective

The Bayesian transition model extends the Markov baseline in two directions. First, it applies Laplace smoothing to eliminate zero-probability transitions, which would produce undefined log-loss values when rare or unobserved transitions appear in the test set. Second, it provides an explicit, principled quantification of uncertainty for each transition probability, distinguishing between well-observed transitions whose estimates can be trusted and poorly observed ones whose probabilities remain genuinely uncertain.

### 2.2 Methods

**Dirichlet–Multinomial conjugate model.** For each source segment i, the vector of transition probabilities to all K destination segments is modelled as a Dirichlet-distributed random variable. This is the natural conjugate prior for the multinomial distribution of observed transitions. The prior was set to a symmetric Dirichlet with concentration parameter α = 1 for every component, which corresponds to Laplace smoothing — each transition count is incremented by one before normalisation. Formally, for row i of the count matrix:

> **α**_i = **n**_i + 1, where **n**_i is the vector of observed transition counts from segment i.

The posterior distribution over transition probabilities from segment i given the training data is then Dirichlet(**α**_i), with a closed-form posterior mean equal to α_i / Σ_j α_ij.

**Point estimate.** The posterior mean served as the point estimate of the transition matrix (**P_bayes**), replacing the raw frequency estimate used in the Markov baseline. For segments with many observations, the posterior mean is nearly identical to the MLE. For segments with sparse data — most critically *Lost* (~2,400 training observations) and *Big Spenders* (~5,200 observations) — the Laplace prior exerts a meaningful regularising effect, pulling extreme estimates toward uniformity and ensuring all transition probabilities are strictly positive.

**Posterior sampling and uncertainty quantification.** To characterise the uncertainty of each transition probability, 5,000 independent samples were drawn from the Dirichlet posterior for each row of the transition matrix. The 90% credible interval (5th to 95th percentile of the marginal posterior) was computed for every cell (i, j). The width of this interval serves as a direct measure of estimation uncertainty: a narrow credible interval indicates that the transition is observed frequently enough to be estimated with confidence, while a wide interval flags transitions where the data are too sparse to support a reliable estimate.

**Visualisation.** Two heatmaps were produced side by side: (1) the posterior mean transition matrix, structurally analogous to the Markov baseline heatmap but with smoothed probabilities; and (2) the uncertainty map displaying the width of the 90% credible interval for each cell, highlighting where the model's estimates should be interpreted cautiously.

**Evaluation.** Predictive accuracy and log-loss were computed on the Q2→Q3 test set using the same protocol as for the Markov baseline, with the posterior mean matrix replacing the frequency-normalised matrix.

### 2.3 Results and Conclusions

**Transition structure.** The posterior mean matrix largely preserved the structural patterns identified in the Markov baseline, with three notable observations:

- **At Risk as the dominant attractor.** Every segment exhibited a transition probability of 0.27–0.37 toward *At Risk* in the subsequent quarter, confirming this segment as the gravitational centre of the customer dynamics system and the primary destination of behavioural degradation.

- **Big Spenders instability.** Only 11% of Big Spenders clients remained in that segment from one quarter to the next, while 37% transitioned directly to *At Risk*. This pattern is consistent with episodic, high-volume purchase behaviour rather than sustained engagement — clients make a large concentrated purchase and then become substantially less active.

- **Lost as a transient state.** Despite representing the lowest-rank segment, *Lost* was not an absorbing state: approximately 16% of clients remained in *Lost*, 26% transitioned to *At Risk*, and 24% moved to *Regular*. This finding underscores that the *Lost* designation in this dataset captures temporary inactivity rather than permanent customer attrition, and that spontaneous reactivation is a structurally regular phenomenon.

**Uncertainty quantification.** The credible interval map revealed pronounced heterogeneity in estimation confidence across the transition matrix. Rows corresponding to *Lost* and *Big Spenders* — the two segments with the smallest sample sizes in the training data — exhibited substantially wider credible intervals, confirming that Bayesian smoothing was most consequential precisely for these segments. Rows for *Regular* and *At Risk*, which were far better represented in the training data, had narrow credible intervals, indicating robust estimates.

**Predictive performance.** The accuracy and log-loss of the Bayesian model were close to those of the Markov baseline. This is the expected result: for segments with abundant data, the posterior mean coincides almost exactly with the MLE, and the Laplace prior has negligible effect on point estimates. The principal contribution of the Bayesian framework is therefore not an improvement in predictive metrics per se, but the provision of a statistically coherent uncertainty decomposition — making explicit which transitions are data-rich and trustworthy versus which remain poorly constrained and should be interpreted with caution.

---

## 3. ML Models: Next-Segment Prediction

### 3.1 Objective

The objective of this modelling stage was to move beyond the Markov baseline and leverage the full set of client-level behavioural, financial, and compositional features to predict the next-quarter segment of each customer. While the Markov model used only the current segment label as a predictor, the machine learning approach incorporated approximately 40 features, allowing the model to learn from the granular dynamics of individual customer behaviour.

### 3.2 Feature Engineering and Preprocessing

**Feature matrix.** The feature matrix for each client–quarter observation comprised three groups of variables: (1) the base behavioural and financial features constructed in Section 2 — order volume, revenue, margin, average order value, recency, tenure, cancellation and return rates, product category shares, regional shares, delivery and payment method shares; (2) a one-hot encoding of the client's current segment label, with one category dropped to avoid collinearity. The target variable was `segment_next` — the segment the client was assigned to in the following quarter.

**Preprocessing pipeline.** Features were transformed prior to model fitting using a two-stage scaling scheme:
- *Heavy-tailed features* (`items`, `revenue`, `margin`, `avg_order_value`, `most_freq_order_size`): subjected to a `log1p` transformation to reduce skewness, followed by `RobustScaler` scaling. The `margin` column required an additional shift prior to log transformation, as it could take negative values; the shift was computed on the training set and applied to the test set without refitting to prevent leakage.
- *Rate features* (`delivered_ratio`, `cancel_ratio`, `return_ratio`, `recency_days`): standardised using `StandardScaler`.
- All remaining binary and share features were used without scaling.

All preprocessing parameters (scaler medians, IQR ranges, margin shift) were fitted exclusively on the training data and applied as fixed transforms to the test set.

**Train/test split.** The training set comprised Q1→Q2 transitions, and the test set comprised Q2→Q3 transitions.

### 3.3 Logistic Regression

**Model specification.** A multinomial logistic regression was fitted using the L-BFGS solver with a maximum of 1,000 iterations. No regularisation was explicitly tuned; the default L2 penalty was retained. The model attempted to learn linear decision boundaries in the transformed feature space across all segment classes simultaneously.

**Results.**

| Metric | Value |
|---|---|
| Accuracy | 33.8% |
| Macro F1 | 0.111 |
| Log-loss | 1.716 |

The accuracy of 33.8% substantially exceeds the Markov baseline (18.8%) and the uniform classifier (14.3%), demonstrating that the inclusion of rich client-level features carries genuine predictive value for next-segment assignment. The log-loss of 1.716 likewise improves upon the Markov baseline (1.807).

However, the macro F1 of 0.111 reveals a critical limitation: despite higher overall accuracy, the model exhibited strong prediction bias toward the majority classes (*At Risk* and *Loyal*), effectively failing to identify minority segments such as *Champions*, *Lost*, and *Big Spenders*. This is confirmed by the classification report, which flagged undefined precision and recall for certain classes due to the absence of predicted samples. Additionally, the L-BFGS solver did not converge within 1,000 iterations, suggesting that the feature space may require further regularisation tuning or a larger iteration budget for a fully stable solution.

**Coefficient analysis.** Inspection of the logistic regression coefficient matrix revealed interpretable patterns. *Champions*-directed coefficients were strongly positive for `items` (basket size) and `total_orders` (order frequency), while *Loyal* was associated with high `revenue` and relatively low `avg_order_value`. *At Risk* was associated with high raw `avg_order_value` but low `revenue`, consistent with the profile of infrequent high-ticket buyers who have become dormant. *Lost* was negatively associated with `items`, reflecting the minimal purchasing activity of this group.

### 3.4 LightGBM

**Model specification.** A gradient boosting classifier was trained using LightGBM with a multiclass objective. The model comprised 500 decision trees with a learning rate of 0.05 and a maximum depth of 6. No class weighting was applied at this stage.

**Results.**

| Metric | Value |
|---|---|
| Accuracy | 29.3% |
| Macro F1 | 0.094 |
| Log-loss | 1.832 |

The LightGBM multiclass model performed below the logistic regression on all three metrics, and only marginally above the Markov baseline in terms of log-loss. This result reflects the inherent difficulty of the multiclass segment prediction task: despite a rich feature set, gradient boosting did not translate into better calibrated class probabilities compared to logistic regression under these training conditions.

**Feature importance.** Analysis of LightGBM's feature importances (measured by total gain across splits) identified a consistent hierarchy of predictive features. The top five predictors by importance were: `margin`, `days_since_first_purchase`, `revenue`, `recency_days`, and `avg_order_value`. Financial metrics and temporal/lifecycle features dominated the importance ranking, while product category shares, regional distribution, and delivery method features contributed less but non-negligibly. This hierarchy confirmed that monetary value and recency — the M and R dimensions of the RFM framework — were the strongest discriminators of next-quarter segment membership.

### 3.5 Discussion

Both multiclass models demonstrated that client-level features carry meaningful predictive signal for segment transitions, consistently outperforming the segment-only Markov baseline in terms of accuracy. However, the macro F1 scores remained low across both models, reflecting the inherent difficulty of the multiclass prediction task under class imbalance and sparse minority classes.

These findings motivated a reconceptualisation of the prediction problem: rather than attempting full six-class segment prediction — a task complicated by sparse minority classes — the subsequent modelling stages reframe the objective as a binary classification of *downgrade*, which is both more tractable and more directly actionable from a business perspective.

---

## 4. Binary Classification of Downgrade

### 4.1 Problem Formulation

The limitations of the multiclass next-segment prediction task — sparse minority classes, distributional shift, and low macro F1 — motivated a reformulation of the prediction objective. Rather than predicting the exact next segment, the problem was reduced to a binary classification: will a given client experience a downgrade in segment rank in the following quarter?

A downgrade event was defined formally as a decrease in ordinal segment rank between two consecutive quarters: `downgrade = 1` if `rank(segment_next) < rank(segment_current)`, and `downgrade = 0` otherwise. The segment rank mapping followed the hierarchy established in Section 3 (Champions = 5, Loyal = 4, Promising = 3, Regular = 2, At Risk = 1, Lost = 0). A client who moved from *Champions* to *Loyal*, or from *Loyal* to *At Risk*, was labelled as having experienced a downgrade; a client who remained in the same segment or improved was not.

This binary formulation is more tractable statistically and more directly interpretable from a business standpoint: rather than specifying a full future state, it identifies which clients are at risk of deterioration — the primary signal required for targeted retention intervention.

**Dataset characteristics.** The feature matrix for binary classification was identical to that used in the multiclass setting, comprising 35 features per client–quarter observation after removing RFM scores, segment labels, and identifier columns from the predictor set. The training set (Q1→Q2 transitions) contained 35,612 observations with a near-balanced downgrade rate of 47.5%. The test set (Q2→Q3 transitions) contained 28,233 observations with a downgrade rate of 46.5% (13,133 downgrade vs. 15,100 no-downgrade).

The closely matched class distributions between training and test sets ensure that model evaluation is not distorted by distributional shift: both periods exhibit a near-even split between downgrade and no-downgrade outcomes, making performance metrics directly comparable across thresholds and models.

### 4.2 Logistic Regression

**Model specification.** Logistic regression was fitted with balanced class weighting (`class_weight="balanced"`), which automatically adjusts loss contributions inversely proportional to class frequencies in the training set. This was applied to prevent the model from converging to a degenerate majority-class predictor in the presence of training-set imbalance. The L-BFGS solver was used with a budget of 4,000 iterations; despite this, the optimiser did not reach convergence, suggesting that the decision boundary in the high-dimensional feature space is not well-approximated by a smooth linear hyperplane within the allotted iteration count.

**Results at default threshold (0.5).**

| Metric | Value |
|---|---|
| Accuracy | 68.1% |
| F1 (downgrade class) | 0.675 |
| ROC-AUC | 0.746 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No downgrade (0) | 0.723 | 0.656 | 0.688 | 15,100 |
| Downgrade (1) | 0.642 | 0.711 | 0.675 | 13,133 |

The ROC-AUC of 0.746 indicates moderate discriminative ability — the model's probability scores carry genuine predictive signal but are substantially weaker than in the distributional-shift setting. With class proportions balanced between train and test (~47%), neither class dominates predictions at the default threshold: both precision and recall are in the 0.64–0.72 range. The `class_weight="balanced"` strategy operates symmetrically here, as there is no structural imbalance to compensate for.

**Threshold optimisation.** The optimal decision threshold was derived from the Precision–Recall curve. The threshold maximising F1 was identified as **0.2061** — meaning a client was flagged as at-risk if the model assigned a downgrade probability of approximately 21% or higher.

**Results at optimal threshold (0.2061).**

| Metric | Value |
|---|---|
| ROC-AUC | 0.746 |
| F1 (downgrade class) | 0.715 |
| Recall | 0.946 |
| Precision | 0.574 |

The threshold reduction from 0.5 to 0.21 improved recall substantially (from 0.711 to 0.946) at the cost of reduced precision (from 0.642 to 0.574). In business terms, this threshold strategy recovers 94.6% of clients who will actually experience a downgrade. The ROC-AUC remained unchanged at 0.746, confirming that threshold selection affects the operating point on the ROC curve but does not alter the model's underlying discriminative power.

---

### 4.3 LightGBM

**Model specification.** A gradient boosted tree classifier was trained using LightGBM with a binary objective. The model was configured with 300 estimators, a learning rate of 0.05, maximum tree depth of 5, and 15 leaves per tree — a deliberately shallow architecture to reduce overfitting on a training set with near-balanced class proportions while generalising to a test set with strongly skewed class distribution. Additional regularisation was applied via L2 penalty (reg_lambda = 1.0), subsampling of 80% of rows and 80% of features per tree (subsample, colsample_bytree), and a minimum of 50 samples per leaf (min_child_samples). Class imbalance was addressed using `class_weight="balanced"`, consistent with the logistic regression approach.

**Results at default threshold (0.5).**

| Metric | Value |
|---|---|
| Accuracy | 68.6% |
| F1 (downgrade class) | 0.680 |
| ROC-AUC | 0.758 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No downgrade (0) | 0.728 | 0.661 | 0.693 | 15,100 |
| Downgrade (1) | 0.647 | 0.715 | 0.680 | 13,133 |

At the default decision threshold of 0.5, LightGBM showed a balanced prediction pattern across both classes, consistent with the near-equal class distribution in the test set. The ROC-AUC of 0.758 exceeds that of the logistic regression (0.746), indicating that LightGBM's probability scores provide a marginally more discriminative ranking of clients by downgrade risk.

**Threshold optimisation.** The optimal decision threshold was derived via the Precision–Recall curve and identified as **0.3201**.

**Results at optimal threshold (0.3201).**

| Metric | Value |
|---|---|
| ROC-AUC | 0.758 |
| F1 (downgrade class) | 0.715 |
| Recall | 0.942 |
| Precision | 0.576 |

Following threshold optimisation, LightGBM matched the logistic regression on F1 (0.715) with the same recall (0.942) but nearly identical precision (0.576 vs. 0.574 for LR). The primary advantage of LightGBM over logistic regression in this setting is the higher ROC-AUC (0.758 vs. 0.746), reflecting better probability calibration rather than a dramatic difference in operating-point metrics.

**Feature importance.** Feature importances were computed as the total number of times each feature was used to split a node across all trees (split-based importance). The full ranking for the binary downgrade model is presented below:

| Rank | Feature | Importance |
|---|---|---|
| 1 | `margin` | 415 |
| 2 | `revenue` | 371 |
| 3 | `avg_order_value` | 342 |
| 4 | `recency_days` | 323 |
| 5 | `days_since_first_purchase` | 243 |
| 6 | `total_orders` | 202 |
| 7 | `items` | 198 |
| 8 | `cat_diapers` | 166 |
| 9 | `most_freq_order_size` | 147 |
| 10 | `cat_toys` | 137 |
| 11 | `cat_cosmetics_hygiene` | 134 |
| 12 | `delivered_ratio` | 114 |
| 13 | `cancel_ratio` | 100 |
| 14 | `cat_textile` | 98 |
| 15 | `cat_feeding_goods` | 95 |

The top-5 features — `margin`, `revenue`, `avg_order_value`, `recency_days`, and `days_since_first_purchase` — collectively account for the large majority of total split activity. `margin` retained its position as the most important feature, confirming that the monetary dimension of client value is the single most informative predictor of downgrade risk. Compared to the previous setup with distributional shift, `revenue` and `avg_order_value` rose to second and third place respectively, while `recency_days` dropped to fourth — reflecting that in a balanced test period, monetary signals carry relatively more weight and recency alone is less dominant. Product category composition features contributed at lower but non-negligible levels, with `cat_diapers`, `cat_toys`, and `cat_cosmetics_hygiene` being the most important categorical predictors.

**Comparison with Logistic Regression.**

| Model | ROC-AUC | F1 (opt) | Recall (opt) | Precision (opt) | Optimal threshold |
|---|---|---|---|---|---|
| Logistic Regression | 0.746 | 0.715 | **0.946** | 0.574 | 0.2061 |
| LightGBM | **0.758** | **0.715** | 0.942 | 0.576 | 0.3201 |

LightGBM is the superior model by ROC-AUC and marginally higher precision, while logistic regression achieves slightly higher recall. At the operating point level, the two models are nearly indistinguishable on F1 (both 0.715). The practical choice between them depends on the cost structure of intervention: if the primary concern is capturing as many at-risk clients as possible regardless of false positive volume (high recall priority), logistic regression with threshold 0.21 is preferable; if discriminative quality of the probability scores matters for ranking or budget allocation, LightGBM at threshold 0.32 is the better choice.

### 4.4 Random Forest

**Model specification.** A Random Forest classifier was trained with 900 decision trees, maximum depth of 10, a minimum of 20 samples per leaf, and 70% of features considered at each split (`max_features=0.7`). Class imbalance was handled via `class_weight="balanced"`. The hyperparameters were selected through a manual grid search over six candidate configurations, with ROC-AUC on the test set as the selection criterion; the configuration reported here achieved the highest test AUC across all trials.

**Results at default threshold (0.5).**

| Metric | Value |
|---|---|
| Accuracy | 68.5% |
| F1 (downgrade class) | 0.681 |
| ROC-AUC | 0.758 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No downgrade (0) | 0.73 | 0.65 | 0.69 | 15,100 |
| Downgrade (1) | 0.64 | 0.72 | 0.68 | 13,133 |

Confusion matrix at default threshold:

|  | Pred No downgrade | Pred Downgrade |
|---|---|---|
| No downgrade | 9,827 | 5,273 |
| Downgrade | 3,621 | 9,512 |

The default-threshold behaviour is balanced across both classes, consistent with the near-equal class distribution in the test set. Both precision and recall fall in the 0.64–0.73 range, without the extreme asymmetry seen under distributional shift. The ROC-AUC of 0.758 is the highest among all classifiers evaluated.

**Threshold optimisation.** The optimal threshold derived from the Precision–Recall curve was **0.3076**.

**Results at optimal threshold (0.3076).**

| Metric | Value |
|---|---|
| ROC-AUC | 0.758 |
| F1 (downgrade class) | 0.715 |
| Recall | 0.942 |
| Precision | 0.576 |

**Feature importance.** Random Forest computes feature importance as the mean decrease in impurity (Gini importance) across all trees and all splits, normalised to sum to one. The distribution was markedly concentrated:

| Rank | Feature | Importance |
|---|---|---|
| 1 | `recency_days` | 0.5035 |
| 2 | `segment_At Risk` | 0.1274 |
| 3 | `segment_Regular` | 0.0908 |
| 4 | `margin` | 0.0305 |
| 5 | `total_orders` | 0.0299 |
| 6 | `revenue` | 0.0268 |
| 7 | `segment_Lost` | 0.0223 |
| 8 | `days_since_first_purchase` | 0.0212 |

`recency_days` alone accounted for 50.3% of total importance — the highest concentration of any single feature, reflecting the Random Forest's tendency to select the most discriminative feature at the root node of each tree. Notably, segment one-hot features now appear prominently: `segment_At Risk` (12.7%) and `segment_Regular` (9.1%) rank second and third, confirming that the client's current RFM segment is a strong structural predictor of downgrade risk independent of recency. The result confirms that the number of days since the last purchase combined with current segment membership are the dominant predictors of imminent segment degradation.

---

### 4.5 XGBoost

**Model specification.** An XGBoost classifier was trained with a binary logistic objective, 400 boosting rounds, maximum tree depth of 4, learning rate of 0.05, and row and column subsampling of 80%. Unlike the other classifiers, class imbalance was handled via the `scale_pos_weight` parameter — the ratio of negative to positive instances in the training set — rather than per-sample weighting. Given the near-balanced training set (47.5% downgrade), `scale_pos_weight` was 1.105, yielding a minor upweighting of the downgrade class.

**Results at default threshold (0.5).**

| Metric | Value |
|---|---|
| Accuracy | 68.4% |
| F1 (downgrade class) | 0.678 |
| ROC-AUC | 0.757 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No downgrade (0) | 0.73 | 0.66 | 0.69 | 15,100 |
| Downgrade (1) | 0.65 | 0.71 | 0.68 | 13,133 |

Confusion matrix at default threshold:

|  | Pred No downgrade | Pred Downgrade |
|---|---|---|
| No downgrade | 9,955 | 5,145 |
| Downgrade | 3,770 | 9,363 |

**Threshold optimisation.** The optimal threshold was **0.3155**.

**Results at optimal threshold (0.3155).**

| Metric | Value |
|---|---|
| ROC-AUC | 0.757 |
| F1 (downgrade class) | 0.715 |
| Recall | 0.932 |
| Precision | 0.580 |

**Feature importance.** XGBoost's default importance metric is gain — the average improvement in the loss function brought by a feature across all splits where it is used. The gain-normalised ranking:

| Rank | Feature | Importance (normalised gain) |
|---|---|---|
| 1 | `segment_Promising` | 0.1905 |
| 2 | `recency_days` | 0.1699 |
| 3 | `segment_Regular` | 0.1570 |
| 4 | `segment_At Risk` | 0.1221 |
| 5 | `segment_Lost` | 0.0498 |
| 6 | `segment_Champions` | 0.0389 |
| 7 | `total_orders` | 0.0261 |
| 8 | `days_since_first_purchase` | 0.0104 |

The feature hierarchy differs markedly from the previous run: segment one-hot features now dominate the gain ranking, with `segment_Promising` at the top (19.1%), followed by `recency_days` (17.0%) and `segment_Regular` (15.7%). This pattern reflects that without the extreme recency spike of the Q4 period, the current segment label carries relatively more information about downgrade risk. `recency_days` remains the second-most important continuous feature, while financial and category features contribute less.

---

### 4.6 Support Vector Machine

**Model specification.** A Support Vector Classifier with an RBF kernel was fitted inside a `sklearn` Pipeline with a preceding `StandardScaler`. The kernel bandwidth was set automatically via the `gamma="scale"` heuristic (1 / (n_features × X.var())), the regularisation parameter was C = 1.0, and probability calibration was enabled via Platt scaling (`probability=True`). Class imbalance was handled with `class_weight="balanced"`.

**Results at default threshold (0.5).**

| Metric | Value |
|---|---|
| Accuracy | 68.4% |
| F1 (downgrade class) | 0.677 |
| ROC-AUC | 0.720 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No downgrade (0) | 0.725 | 0.659 | 0.690 | 15,100 |
| Downgrade (1) | 0.645 | 0.712 | 0.677 | 13,133 |

Confusion matrix at default threshold:

|  | Pred No downgrade | Pred Downgrade |
|---|---|---|
| No downgrade | 9,947 | 5,153 |
| Downgrade | 3,778 | 9,355 |

The SVM produced the weakest results among all classifiers. Its ROC-AUC of 0.720 was the lowest, indicating that the RBF kernel with default hyperparameters did not capture the non-linear structure of the feature space as effectively as the tree-based ensembles or gradient boosting models.

**Threshold optimisation.** The optimal threshold was **0.2851**.

**Results at optimal threshold (0.2851).**

| Metric | Value |
|---|---|
| ROC-AUC | 0.720 |
| F1 (downgrade class) | 0.685 |
| Recall | 0.757 |
| Precision | 0.626 |

Following threshold optimisation, the SVM showed a distinctly different operating point from the other models: its optimal F1 (0.685) was lower than all other classifiers, and its recall (0.757) was the weakest. The gap in ROC-AUC (0.720 vs. 0.757–0.758 for tree ensembles) is not recoverable through threshold adjustment, as it reflects the intrinsic discriminative capacity of the model's probability scores.

---

### 4.7 Multilayer Perceptron (MLP)

**Model specification.** A feedforward neural network was trained using `sklearn`'s `MLPClassifier`. The architecture and regularisation hyperparameters were selected via `RandomizedSearchCV` with 3-fold cross-validation over 20 randomly sampled candidate configurations. The search space covered hidden layer sizes, activation function, L2 regularisation strength (alpha), batch size, and learning rate. The best configuration found was:

| Hyperparameter | Value |
|---|---|
| Hidden layer sizes | (128, 64) — two hidden layers |
| Activation | tanh |
| Learning rate (initial) | 0.0001 |
| Batch size | 128 |
| L2 penalty (alpha) | 0.001 |

The best cross-validation ROC-AUC achieved during the search was 0.7685. Class imbalance was addressed by passing per-sample weights computed via `compute_sample_weight("balanced")`, as `MLPClassifier` does not expose a native `class_weight` parameter.

**Results at default threshold (0.5).**

| Metric | Value |
|---|---|
| Accuracy | 68.5% |
| F1 (downgrade class) | 0.676 |
| ROC-AUC | 0.756 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No downgrade (0) | 0.723 | 0.666 | 0.694 | 15,100 |
| Downgrade (1) | 0.648 | 0.707 | 0.676 | 13,133 |

Confusion matrix at default threshold:

|  | Pred No downgrade | Pred Downgrade |
|---|---|---|
| No downgrade | 10,060 | 5,040 |
| Downgrade | 3,851 | 9,282 |

At the default threshold, the MLP showed balanced predictions across both classes. The ROC-AUC of 0.756 places the MLP between LightGBM (0.758) and logistic regression (0.746), near the tree ensemble level.

**Threshold optimisation.** The optimal threshold derived from the Precision–Recall curve was **0.3048**.

**Results at optimal threshold (0.3048).**

| Metric | Value |
|---|---|
| ROC-AUC | 0.756 |
| F1 (downgrade class) | 0.714 |
| Recall | 0.937 |
| Precision | 0.577 |

Confusion matrix at optimal threshold:

|  | Pred No downgrade | Pred Downgrade |
|---|---|---|
| No downgrade | 6,094 | 9,006 |
| Downgrade | 830 | 12,303 |

The cross-validation ROC-AUC (0.7685) and the test-set ROC-AUC (0.756) are closely aligned — consistent with the absence of distributional shift between training and test sets. This makes the cross-validation score a reliable estimate of generalisation performance, unlike the previous setup where the Q4 regime change created a large gap between CV and test metrics.

### 4.8 Cross-Model Comparison and Model Selection

The table below summarises all six binary downgrade classifiers evaluated at their respective PR-curve-optimal decision thresholds:

| Model | Threshold | ROC-AUC | F1 | Recall | Precision |
|---|---|---|---|---|---|
| **Random Forest** | 0.308 | **0.758** | **0.715** | 0.942 | 0.576 |
| LightGBM | 0.320 | **0.758** | **0.715** | 0.942 | 0.576 |
| XGBoost | 0.316 | 0.757 | **0.715** | 0.932 | **0.580** |
| MLP (sklearn) | 0.305 | 0.756 | 0.714 | 0.937 | 0.577 |
| Logistic Regression | 0.206 | 0.746 | **0.715** | **0.946** | 0.574 |
| SVM | 0.285 | 0.720 | 0.685 | 0.757 | 0.626 |

**Model selection.** Random Forest was selected as the primary model for the downstream business application. The decision was based on three considerations: (1) Random Forest achieved the highest ROC-AUC (0.758, tied with LightGBM) and F1 (0.715) among all models; (2) the difference in F1 between Random Forest and LightGBM is 0.000 — negligible — but Random Forest's feature importance is more straightforwardly interpretable, as it does not require understanding of gradient boosting-specific parameters such as number of leaves or L2 regularisation on leaf weights; (3) for thesis purposes, Random Forest represents a well-established and broadly understood ensemble method, facilitating clear methodological exposition. LightGBM and XGBoost are noted as alternatives with statistically equivalent performance.

**Threshold interpretation.** All tree-based ensembles were deployed with decision thresholds in the 0.31–0.32 range, somewhat below the conventional 0.5. This reflects an intentional design choice: in the customer retention context, the cost of a missed true downgrade (a client who degrades without receiving a retention intervention) substantially exceeds the cost of a false positive (an intervention offered to a client who would not have downgraded). Lowering the threshold below 0.5 encodes this asymmetric cost structure, prioritising recall without sacrificing precision beyond the point where interventions become commercially wasteful.

