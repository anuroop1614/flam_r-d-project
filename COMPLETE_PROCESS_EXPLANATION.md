# Complete Process and Steps: Parametric Curve Fitting

## Executive Summary

This project implements a parametric curve fitting algorithm to model 2D data points using a rotated coordinate system with an exponentially modulated sinusoidal oscillation. The model fits three parameters (rotation angle θ, horizontal shift X, and exponential modulation M) using a two-stage approach: initial analytical estimates followed by grid search refinement.

---

## Problem Statement

Given a set of (x, y) data points, we aim to find a parametric curve that best fits the data. The curve follows a rotated straight line with a perpendicular oscillating component whose amplitude varies exponentially.

---

## Model Formulation

### Parametric Equations

The curve is parameterized by variable `t`:

```
x(t) = t·cos(θ) - e^(M|t|)·sin(0.3t)·sin(θ) + X
y(t) = 42 + t·sin(θ) + e^(M|t|)·sin(0.3t)·cos(θ)
```

### Parameters

1. **θ (theta)**: Rotation angle in radians (fitted in degrees: 0.1° to 49.9°)
   - Controls the main direction of the curve
   
2. **X**: Horizontal shift parameter (0 to 100)
   - Translates the curve horizontally
   
3. **M**: Exponential modulation parameter (-0.05 to 0.05)
   - Controls exponential growth (M > 0) or decay (M < 0) of oscillation amplitude
   - Fixed frequency: 0.3 radians per unit t

### Coordinate Transformation

In a rotated coordinate system:
- **t-coordinate** (along main direction): `t = (x - X)·cos(θ) + (y - 42)·sin(θ)`
- **v-coordinate** (perpendicular): `v = -(x - X)·sin(θ) + (y - 42)·cos(θ)`

The model constraint: `v(t) = e^(M|t|)·sin(0.3t)`

---

## Step-by-Step Process

### Step 1: Data Loading and Preprocessing

```python
df = pd.read_csv("xy_data.csv")
x = df[df.columns[0]].to_numpy().astype(float)
y = df[df.columns[1]].to_numpy().astype(float)
```

- Load CSV file with x, y columns (case-insensitive)
- Convert to numpy arrays for efficient computation
- Total: 1501 data points

---

### Step 2: Initial Parameter Estimation

#### 2.1 Estimate θ (Rotation Angle)

**Method**: Linear Regression

```python
def fit_theta_linear(x, y):
    # Fit: y = a + b·x using least squares
    X = np.vstack([np.ones_like(x), x]).T
    b = np.linalg.lstsq(X, y, rcond=None)[0][1]  # slope
    theta = np.degrees(np.arctan2(b, 1.0))
    return float(np.clip(theta, 0.1, 49.9))
```

**Rationale**: 
- The main direction of the curve should align with the overall trend of the data
- Linear regression gives the slope, which corresponds to tan(θ)
- Clamp to valid range (0.1° to 49.9°)

**Result**: θ₀ (initial estimate)

---

#### 2.2 Estimate X (Horizontal Shift)

**Method**: Centering the t-range

```python
def choose_X_for_t_range(x, y, theta, target_mean=33.0):
    th = deg2rad(theta)
    ct, st = np.cos(th), np.sin(th)
    # Calculate t values assuming X=0
    t0 = x*ct + (y-42.0)*st
    mean_t0 = float(np.median(t0))
    # Adjust X to center t-range around target
    X = (mean_t0 - target_mean) / ct
    return float(np.clip(X, 0.0, 100.0))
```

**Rationale**:
- X shifts the t-coordinate: `t_new = t_old - X·cos(θ)`
- By choosing X appropriately, we center the t-range around a target mean (33.0)
- This ensures the curve spans the appropriate parameter range
- Use median (robust to outliers) instead of mean

**Result**: X₀ (initial estimate)

---

#### 2.3 Estimate M (Exponential Modulation)

**Method**: Log-Linear Regression

```python
def fit_M(t, v):
    # Filter points where |sin(0.3t)| is significant
    s = np.sin(0.3 * t)
    mask = np.abs(s) > 0.2
    
    # Model: v = e^(M|t|)·sin(0.3t)
    # Taking log: log(|v/sin(0.3t)|) = M·|t|
    z = np.log(np.clip(np.abs(v[mask] / s[mask]), 1e-12, None))
    w = np.abs(t[mask])
    
    # Linear least squares: z = M·w (no intercept)
    M = np.sum(w * z) / np.sum(w*w)
    return float(np.clip(M, -0.05, 0.05))
```

**Rationale**:
- From the model: `v = e^(M|t|)·sin(0.3t)`
- Taking absolute value and dividing by sin(0.3t): `|v/sin(0.3t)| = e^(M|t|)`
- Taking logarithm: `log(|v/sin(0.3t)|) = M·|t|`
- Filter points where sin(0.3t) ≈ 0 to avoid division by zero
- Linear regression without intercept gives M
- Clamp to valid range (-0.05 to 0.05)

**Result**: M₀ (initial estimate)

---

### Step 3: Coordinate Transformation and Projection

```python
def rotate_and_project(x, y, theta_deg, X):
    th = deg2rad(theta_deg)
    ct, st = math.cos(th), math.sin(th)
    
    # Shift by (X, 42)
    Xc = x - X
    Yc = y - 42.0
    
    # Rotate to (t, v) coordinates
    t =  Xc*ct + Yc*st  # along main direction
    v = -Xc*st + Yc*ct  # perpendicular (should match e^(M|t|)sin(0.3t))
    
    return t, v
```

**Purpose**: Transform data from (x, y) to rotated coordinate system (t, v) for model evaluation.

---

### Step 4: Loss Function (L1 Normal Misfit)

```python
def l1_normal_misfit(theta, X, M, x, y):
    t, v = rotate_and_project(x, y, theta, X)
    pred = np.exp(np.abs(t)*M) * np.sin(0.3*t)  # predicted v
    return float(np.mean(np.abs(v - pred)))     # mean absolute error
```

**Rationale**:
- L1 (mean absolute error) is robust to outliers
- Focus on perpendicular direction (v) where the model constraint applies
- Minimize: `mean(|v_actual - v_predicted|)`

---

### Step 5: Grid Search Refinement

**Method**: 3D Grid Search with Adaptive M-refinement

```python
def refine(theta0, X0, M0, x, y):
    # Define search grids
    theta_grid = np.linspace(theta0-3°, theta0+3°, 31 points)
    X_grid     = np.linspace(X0-5, X0+5, 41 points)
    
    best = (inf, theta0, X0, M0)
    
    for theta in theta_grid:
        for X in X_grid:
            # Re-fit M for this (theta, X) pair
            t, v = rotate_and_project(x, y, theta, X)
            M_guess = fit_M(t, v)
            
            # Fine search around M_guess
            for M in np.linspace(M_guess-0.01, M_guess+0.01, 21 points):
                misfit = l1_normal_misfit(theta, X, M, x, y)
                if misfit < best[0]:
                    best = (misfit, theta, X, M)
    
    return best
```

**Strategy**:
1. **Coarse grid** for θ and X (±3° and ±5 units)
2. **Adaptive M-refinement**: For each (θ, X) pair:
   - Re-fit M using analytical method
   - Search ±0.01 around that estimate (fine grid)
3. **Total evaluations**: ~31 × 41 × 21 ≈ 26,691 combinations
4. **Selection**: Choose parameters with minimum L1 normal misfit

**Rationale**:
- Grid search ensures global optimum in search region
- Re-fitting M adaptively reduces search space
- Fine grid around M_guess captures local optimum
- Computationally efficient compared to full 3D grid

---

### Step 6: L1 Distance Metric Calculation

**Purpose**: Evaluate fit quality using uniformly sampled points

```python
def calculate_l1_distance_uniform_samples(theta, X, M, x_data, y_data, n_samples=1000):
    # 1. Determine valid t-range from data
    t_data, _ = rotate_and_project(x_data, y_data, theta, X)
    t_min, t_max = np.min(t_data), np.max(t_data)
    
    # 2. Uniformly sample t values
    t_samples = np.linspace(t_min, t_max, n_samples)
    
    # 3. Generate predicted (x, y) from model
    x_pred = t_samples*cos(θ) - e^(M|t|)·sin(0.3t)·sin(θ) + X
    y_pred = 42 + t_samples*sin(θ) + e^(M|t|)·sin(0.3t)·cos(θ)
    
    # 4. For each predicted point, find nearest data point
    # 5. Calculate L1 (Manhattan) distance: |x_pred - x_data| + |y_pred - y_data|
    # 6. Return statistics: mean, median, total, max, std
```

**Metric**: Mean L1 distance between uniformly sampled predicted points and nearest actual data points.

---

## Algorithm Flowchart

```
START
  ↓
Load Data (xy_data.csv)
  ↓
Estimate θ₀ (linear regression)
  ↓
Estimate X₀ (center t-range)
  ↓
Project to (t, v) coordinates
  ↓
Estimate M₀ (log-linear regression)
  ↓
Grid Search Refinement:
  ├─ For each θ in [θ₀-3°, θ₀+3°]:
  │   ├─ For each X in [X₀-5, X₀+5]:
  │   │   ├─ Project data to (t, v)
  │   │   ├─ Re-fit M
  │   │   └─ For each M in [M_guess-0.01, M_guess+0.01]:
  │   │       ├─ Calculate L1 normal misfit
  │   │       └─ Update best if better
  │   └─
  └─
  ↓
Output: Best (θ, X, M)
  ↓
Calculate L1 distance metric
  ↓
Generate LaTeX/Desmos equation
  ↓
Visualize results
  ↓
END
```

---

## Key Design Decisions

### 1. Why L1 Loss Instead of L2?

- **Robustness**: L1 (mean absolute error) is less sensitive to outliers
- **Interpretability**: L1 directly measures average deviation
- **Perpendicular focus**: We care about deviation in the v-direction, not overall Euclidean distance

### 2. Why Two-Stage Approach?

- **Speed**: Initial analytical estimates provide good starting point
- **Accuracy**: Grid search refines parameters near optimum
- **Robustness**: Analytical estimates are stable; grid search handles non-convexity

### 3. Why Adaptive M-Refinement?

- **Efficiency**: M depends strongly on (θ, X), so re-fitting reduces search space
- **Accuracy**: Fine grid around re-fitted M captures local optimum
- **Computational**: Reduces from 31×41×41 = 52,171 to ~26,691 evaluations

### 4. Why Median for X Estimation?

- **Robustness**: Median is less sensitive to outliers than mean
- **Stability**: Provides reliable center estimate even with noisy data

### 5. Why Filter sin(0.3t) for M Estimation?

- **Numerical stability**: Avoid division by zero when sin(0.3t) ≈ 0
- **Signal quality**: Points near zeros don't contribute useful information
- **Threshold**: |sin(0.3t)| > 0.2 ensures meaningful signal

---

## Computational Complexity

- **Initial estimates**: O(n) where n = number of data points
- **Grid search**: O(n × N_θ × N_X × N_M) where N_θ=31, N_X=41, N_M≈21
- **Total**: O(n × 26,691) ≈ O(40 million) operations for n=1501
- **Runtime**: ~1-5 seconds on modern hardware

---

## Validation and Quality Metrics

### 1. L1 Normal Misfit
- **Definition**: Mean absolute error in perpendicular (v) direction
- **Range**: Lower is better (typically 0.01 to 1.0 for good fits)
- **Interpretation**: Average deviation from model constraint

### 2. L1 Distance Metric
- **Definition**: Mean L1 distance between uniformly sampled predicted points and nearest data points
- **Range**: Lower is better
- **Interpretation**: Overall fit quality in (x, y) space

### 3. Visual Inspection
- Overlay predicted curve on data
- Check for systematic biases
- Verify oscillation pattern matches

---

## Limitations and Assumptions

### Assumptions:
1. **Fixed frequency**: Oscillation frequency is 0.3 radians/unit (not fitted)
2. **Fixed offset**: Vertical offset is 42 (not fitted)
3. **Parametric form**: Data follows the specified parametric equation
4. **No outliers**: Robust methods (L1, median) help but assume most points are valid

### Limitations:
1. **Local optimum**: Grid search finds best in search region, not global optimum
2. **Parameter ranges**: Hard constraints (0.1°≤θ≤49.9°, etc.) may limit fit quality
3. **Uniform sampling**: L1 metric assumes uniform t-sampling is appropriate
4. **Nearest neighbor**: L1 metric uses nearest neighbor, which may not be optimal

---

## Potential Improvements

1. **Global optimization**: Use gradient-based methods (e.g., scipy.optimize) for refinement
2. **Adaptive frequency**: Fit oscillation frequency if it's not fixed
3. **Outlier detection**: Identify and handle outliers before fitting
4. **Cross-validation**: Split data to validate generalization
5. **Uncertainty quantification**: Estimate parameter confidence intervals
6. **Multi-start optimization**: Try multiple initial guesses to avoid local minima

---

## Conclusion

This implementation successfully fits a parametric curve to 2D data using a two-stage approach: analytical initial estimates followed by grid search refinement. The model captures both the main directional trend and the oscillatory perpendicular component with exponential amplitude modulation. The L1-based loss functions provide robust fitting, and the uniform sampling metric evaluates overall fit quality.

---

## References

- Least Squares Regression: Standard linear algebra approach
- Grid Search: Exhaustive search over parameter space
- L1 Norm: Manhattan distance, robust to outliers
- Coordinate Rotation: Standard 2D rotation transformation

