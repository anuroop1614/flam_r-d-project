# Parametric Curve Fitting Model Explanation

## Overview
This code fits a 2D parametric curve to (x,y) data points. The model represents a curve that follows a rotated straight line with a perpendicular oscillating component that has an exponentially modulated amplitude.

## The Model Equation

The parametric curve is defined by the parameter `t` with the following equations:

### Parametric Form:
```
x(t) = t·cos(θ) - e^(M|t|)·sin(0.3t)·sin(θ) + X
y(t) = 42 + t·sin(θ) + e^(M|t|)·sin(0.3t)·cos(θ)
```

Where:
- `t` is the parameter (typically ranges from ~6 to ~60)
- `θ` (theta) is the rotation angle in radians (fitted as degrees, 0-50°)
- `X` is the horizontal shift/offset
- `M` is the exponential growth/decay parameter (-0.05 to 0.05)
- `42` is a fixed vertical offset

### Vector Form:
In a rotated coordinate system:
- **Main direction (t)**: Follows a straight line at angle θ
  - `t = (x - X)·cos(θ) + (y - 42)·sin(θ)`
  
- **Perpendicular direction (v)**: Oscillates with exponential envelope
  - `v = e^(M|t|)·sin(0.3t)`
  - This represents the deviation perpendicular to the main direction

## Model Parameters

1. **θ (theta)**: Rotation angle of the main direction
   - Range: 0.1° to 49.9°
   - Initial estimate: From linear regression slope of y vs x
   
2. **X**: Horizontal shift parameter
   - Range: 0.0 to 100.0
   - Initial estimate: Chosen to center the t-range around mean ≈ 33
   
3. **M**: Exponential modulation parameter
   - Range: -0.05 to 0.05
   - Controls exponential growth (M > 0) or decay (M < 0) of oscillation amplitude
   - Initial estimate: Fitted from log-linear relationship

## Fitting Process

### Step 1: Initial Estimates
1. **fit_theta_linear()**: 
   - Performs linear regression: y = a + b·x
   - Calculates θ = arctan(slope)
   - Clamps to (0.1°, 49.9°)

2. **choose_X_for_t_range()**:
   - Rotates data by θ
   - Finds median t-value
   - Adjusts X to center t-range around target mean (33.0)

3. **fit_M()**:
   - Projects data to (t, v) coordinates
   - Filters points where |sin(0.3t)| > 0.2 (to avoid division issues)
   - Fits: log(|v/sin(0.3t)|) ≈ M·|t|
   - Uses linear least squares without intercept

### Step 2: Refinement (Grid Search)
The `refine()` function performs a 3D grid search:
- **θ grid**: ±3° around initial estimate, 31 points
- **X grid**: ±5 units around initial estimate, 41 points  
- **M grid**: For each (θ, X) pair:
  1. Re-fit M using fit_M()
  2. Search ±0.01 around that M, 21 points
  3. Evaluate L1 misfit for each combination

### Step 3: Loss Function
**L1 Normal Misfit**: 
- Projects data to (t, v) coordinates
- Predicts: v_pred = e^(M|t|)·sin(0.3t)
- Calculates: mean(|v - v_pred|)
- Minimizes this L1 loss (mean absolute error in the perpendicular direction)

## Key Functions

### `rotate_and_project(x, y, theta_deg, X)`
Transforms (x,y) to rotated coordinates (t,v):
- Shifts: (x - X, y - 42)
- Rotates by θ: 
  - `t = (x-X)·cos(θ) + (y-42)·sin(θ)`
  - `v = -(x-X)·sin(θ) + (y-42)·cos(θ)`

### `l1_normal_misfit(theta, X, M, x, y)`
Evaluates how well the model fits:
- Projects data to (t,v)
- Compares v with predicted e^(M|t|)·sin(0.3t)
- Returns mean absolute error

## Output

The script outputs:
1. Initial parameter estimates
2. Refined parameters after grid search
3. LaTeX/Desmos-compatible equation string
4. Visualization plot showing data points vs fitted curve

## Physical Interpretation

This model is suitable for data that:
- Follows a roughly linear trend (main direction t)
- Has oscillatory deviations perpendicular to that trend
- Has oscillations whose amplitude grows or decays exponentially with distance along the main direction
- The frequency of oscillation is fixed (0.3 radians per unit t)

Common applications: spiral patterns, wave guides, oscillatory trajectories with amplitude modulation, etc.

