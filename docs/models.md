# TRADY Machine Learning Model Layer

## 1. Overview & Research Scope

The TRADY Machine Learning Model Layer provides a disciplined statistical research framework for evaluating whether machine-learning models can identify repeatable patterns in historical financial-market data.

### Critical Research Guardrails
- **Educational & Research Focus**: TRADY is strictly an analytical and paper-trading platform. It does not provide trading advice, does not claim guaranteed profits, and does not execute real-money transactions.
- **No Direct Execution**: Model outputs are strictly probabilistic or continuous forecasts, never automated buy/sell orders.
- **Independence**: The model layer consumes features ($X$) from the Feature Engine and forward outcomes ($y$) from the Target Engine via the `ModelingDataset` abstraction.

---

## 2. Temporal Discipline: Chronological Splitting

Financial time series exhibit strong temporal autocorrelation. Random train/test shuffling (such as standard k-fold cross-validation or `train_test_split(shuffle=True)`) introduces catastrophic lookahead bias: future information bleeds into training folds, producing artificially inflated backtest performance.

```
Time ─────────────────────────────────────────────────────────────►
[      Train Partition      ] [Gap] [  Validation  ] [Gap] [    Test    ]
   t=0 ─────────────► t=T_train        t=T_val1 ─► t=T_val2    t=T_test ─► t=N
```

### Invariants Enforced by `ChronologicalSplitter`:
1. **Strict Temporal Monotonicity**: $t_{\text{train}} < t_{\text{val}} < t_{\text{test}}$.
2. **No Shuffling**: Records are sorted ascending by timestamp prior to index assignment.
3. **Purge Gap / Horizon Embargo**: When predicting a forward target with horizon $H$ (e.g., 5-day return), the final $H$ bars of the training set represent future intervals that overlap with the start of the validation period. Specifying `purge_gap = H` automatically discards these boundary bars, preventing target bleed across folds.

---

## 3. Supported Model Architectures

| Architecture | Identifier | Description | Key Parameters |
| :--- | :--- | :--- | :--- |
| **Naive Baseline** | `baseline` | Minimal benchmark (majority class or historical mean/zero return) | `strategy` (`majority_class`, `zero`, `mean`) |
| **Logistic Regression** | `logistic_regression` | L2-regularized linear classifier with in-sample standardization | `C`, `penalty`, `solver`, `scale_features` |
| **Ridge Regression** | `ridge_regression` | L2-regularized linear regressor with in-sample standardization | `alpha`, `scale_features` |
| **Random Forest** | `random_forest` | Bagged decision tree ensemble with Gini importance tracking | `n_estimators`, `max_depth`, `min_samples_split` |
| **LightGBM** | `lightgbm` | Fast gradient boosted trees with validation early stopping | `n_estimators`, `learning_rate`, `num_leaves` |
| **XGBoost** | `xgboost` | Regularized gradient boosted decision trees | `n_estimators`, `learning_rate`, `max_depth` |

---

## 4. Evaluation Metrics

### Binary Classification
- **Accuracy**: $\frac{\text{TP} + \text{TN}}{\text{Total}}$
- **Precision**: $\frac{\text{TP}}{\text{TP} + \text{FP}}$
- **Recall**: $\frac{\text{TP}}{\text{TP} + \text{FN}}$
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve
- **Brier Score**: Mean squared error between predicted probabilities and outcomes: $\frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2$ (calibration metric)
- **Log Loss**: Cross-entropy calibration metric

### Continuous Regression
- **Mean Absolute Error (MAE)**: $\frac{1}{N} \sum |y_i - \hat{y}_i|$
- **Root Mean Squared Error (RMSE)**: $\sqrt{\frac{1}{N} \sum (y_i - \hat{y}_i)^2}$
- **$R^2$ Score**: Proportion of variance explained
- **Directional Accuracy**: $\frac{1}{N} \sum \mathbb{I}(\text{sign}(\hat{y}_i) == \text{sign}(y_i))$ (percentage of correct return sign forecasts)
- **Pearson Correlation**: Linear correlation between predicted and actual returns

---

## 5. Schema Validation & Audit Provenance

Every trained model verifies:
1. **Feature Schema**: Number of features, feature names, and feature ordering must match the training set exactly. Any missing or extra column raises `ValueError`.
2. **Target Schema**: Target types and binary labels $\{0, 1\}$ are verified before fitting.
3. **Audit Metadata**: Serialized with `models/<model_id>.joblib` is `models/<model_id>.joblib.meta.json` containing:
   - Unique `experiment_id` and `model_id`
   - Model architecture and hyperparameters
   - Ordered list of training `feature_names`
   - Target definition (name, horizon, threshold, method)
   - Temporal partition boundaries (`train`, `validation`, `test`)
   - Complete software versions (Python, scikit-learn, LightGBM, XGBoost, etc.)
   - Full evaluation metrics across all folds

---

## 6. CLI Usage

```powershell
# 1. List supported architectures
trady models list-types

# 2. Train a model on a labeled dataset
trady models train data/targets/SPY_1d_labeled.parquet --model lightgbm --target target_binary_up_5d --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 -o models/

# 3. Inspect trained model metadata and metrics
trady models inspect models/lightgbm_target_binary_up_5d_abc123.joblib

# 4. Evaluate an existing model on a dataset
trady models evaluate models/lightgbm_target_binary_up_5d_abc123.joblib data/targets/SPY_1d_labeled.parquet
```
