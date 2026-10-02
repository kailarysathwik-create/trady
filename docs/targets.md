# TRADY Target Engine: Label Engineering & Leakage Prevention

## 1. Purpose & Separation Principle

The **TRADY Target Engine** (`src/trady/targets/`) creates forward-looking outcomes (labels) for supervised quantitative machine-learning experiments.

In financial quantitative research, a foundational mistake is **target leakage** (mixing future target data into prediction features). TRADY establishes a strict, architectural separation:

```
FEATURES (X)  ->  Information available at prediction time:  t <= T
TARGETS  (y)  ->  Future outcome being predicted:           t > T
```

> [!CRITICAL]
> **Strict Disjointness Invariant:**
> The set of feature columns and target columns must be strictly disjoint:
> $$\text{Features} \cap \text{Targets} = \emptyset$$
> Target values must **NEVER** be accessible or included in model input feature matrices.

---

## 2. Mathematical Target Definitions

### A. Forward Return ($R^{\text{fwd}}_{t, H}$)
The future price return realized over the forward horizon $H$ bars:

$$R^{\text{fwd}}_{t, H} = \frac{P_{t+H} - P_t}{P_t}$$

$$r^{\text{fwd}}_{t, H} = \ln\left(\frac{P_{t+H}}{P_t}\right)$$

### B. Binary Classification Target ($Y^{\text{binary}}_{t, H, \theta}$)
Predicts whether the future return over horizon $H$ exceeds a decision boundary threshold $\theta$:

$$Y^{\text{binary}}_{t, H, \theta} = \begin{cases}
  1.0 & \text{if } R^{\text{fwd}}_{t, H} > \theta \\
  0.0 & \text{if } R^{\text{fwd}}_{t, H} \le \theta \\
  \text{NaN} & \text{if } t > N - 1 - H
\end{cases}$$

Standard thresholds:
- Directional Up/Down: $\theta = 0.0$
- Large Move Outperform: $\theta = 0.01$ (+1%)

### C. Future Volatility ($\sigma^{\text{fwd}}_{t, H}$)
The sample standard deviation of single-period returns over the future window $[t+1 \dots t+H]$, annualized:

$$\sigma^{\text{fwd}}_{t, H} = \sqrt{\frac{1}{H-1} \sum_{i=1}^{H} (r_{t+i} - \bar{r}_{\text{fwd}})^2}$$

$$\text{RV}^{\text{fwd}}_{t, H} = \sigma^{\text{fwd}}_{t, H} \times \sqrt{252}$$

---

## 3. Unavailable Trailing Targets (End-of-Sample Boundary)

For any bar at index $t$ where $t > N - 1 - H$, the future bar $t+H$ has not occurred yet. 

TRADY enforces:
1. **Explicit NaN:** The final $H$ observations of any horizon-$H$ target are strictly assigned `NaN`. They are never imputed with past data or defaulted to 0.
2. **ModelingDataset Alignment:** [`ModelingDataset.get_modeling_arrays()`](file:///s:/Trady/src/trady/targets/dataset.py#L90-L135) automatically drops warmup rows (where features are NaN) and trailing rows (where targets are NaN), returning a pristine, aligned $(X, y)$ dataset.

---

## 4. ModelingDataset Architecture

[`ModelingDataset`](file:///s:/Trady/src/trady/targets/dataset.py#L38-L150) pairs features and forward targets while preventing leakage:

```python
dataset = ModelingDataset(
    table=combined_table,
    feature_names=("return_1d", "vol_std_10d", "dist_high_10d"),
    target_names=("target_fwd_ret_5d", "target_binary_up_5d"),
)

# Extract feature matrix X (guaranteed to contain 0 target columns)
X = dataset.get_X(drop_na=True)

# Extract target vector y
y = dataset.get_y("target_binary_up_5d", drop_na=True)

# Extract paired aligned arrays
X, y, timestamps, symbols = dataset.get_modeling_arrays("target_fwd_ret_5d")
```

---

## 5. CLI Usage

- `trady targets list`: List registered targets, types, and horizons.
- `trady targets compute <dataset> [--features <feat_file>] [-o <output>]`: Compute forward targets and assemble modeling dataset.
- `trady targets inspect <dataset>`: Inspect class distributions and trailing NaN counts.
