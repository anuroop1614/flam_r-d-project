# Parametric Curve Fitting Project

Fitting a 2D parametric curve to data points using a rotated coordinate system with exponentially modulated sinusoidal oscillations.

## 📋 Overview

This project implements an algorithm to fit a parametric curve of the form:

```
x(t) = t·cos(θ) - e^(M|t|)·sin(0.3t)·sin(θ) + X
y(t) = 42 + t·sin(θ) + e^(M|t|)·sin(0.3t)·cos(θ)
```


# Parametric Curve Fitting Project

Fitting a 2D parametric curve to data points using a rotated coordinate system with exponentially modulated sinusoidal oscillations.

... (content trimmed; full content will be inserted) ...

## ✅ Final Fitted Equation

$$
\left(
t\cdot\cos(0.5227505495584853)

* e^{0.029972582162542474|t|} \cdot \sin(0.3t) \cdot \sin(0.5227505495584853)

- 55.01160286897831,

42 + t\cdot\sin(0.5227505495584853)

* e^{0.029972582162542474|t|} \cdot \sin(0.3t) \cdot \cos(0.5227505495584853)
  \right)
  $$

Where the parameters θ (rotation angle), X (horizontal shift), and M (exponential modulation) are estimated from data.

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Usage

1. Place your data file `xy_data.csv` in the project directory (columns: `x, y`)
2. Run the fitting algorithm:

```bash
python fit_parametric_curve.py
```

### Output

The script outputs:
- Initial parameter estimates
- Refined parameters after grid search
- L1 distance metrics (mean, median, total, max)
- LaTeX/Desmos-compatible equation string
- Visualization plot

## 📁 Project Structure

```
.
├── fit_parametric_curve.py    # Main fitting algorithm
├── xy_data.csv                # Input data file
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── MODEL_EXPLANATION.md       # Detailed model explanation
├── COMPLETE_PROCESS_EXPLANATION.md  # Step-by-step process
└── EQUATIONS.txt              # Equation reference
```

## 🔧 Algorithm Overview

### Two-Stage Fitting Process

1. **Initial Estimates**:
   - **θ (theta)**: Linear regression to estimate rotation angle
   - **X**: Center t-range around target mean
   - **M**: Log-linear regression on exponential envelope

2. **Grid Search Refinement**:
   - Search ±3° around θ₀ (31 points)
   - Search ±5 units around X₀ (41 points)
   - For each (θ, X), re-fit M and search ±0.01 (21 points)
   - Minimize L1 normal misfit

### Key Functions

- `fit_theta_linear()`: Estimate rotation angle from linear regression
- `choose_X_for_t_range()`: Estimate horizontal shift to center parameter range
- `fit_M()`: Estimate exponential modulation parameter
- `rotate_and_project()`: Transform (x,y) to rotated coordinates (t,v)
- `l1_normal_misfit()`: Calculate L1 loss in perpendicular direction
- `refine()`: Grid search refinement
- `calculate_l1_distance_uniform_samples()`: Evaluate fit quality

## 📊 Metrics

### L1 Normal Misfit
Mean absolute error in the perpendicular (v) direction:
```
misfit = mean(|v_actual - e^(M|t|)·sin(0.3t)|)
```

### L1 Distance Metric
Mean L1 (Manhattan) distance between uniformly sampled predicted points and nearest data points:
```
L1_distance = mean(|x_pred - x_data| + |y_pred - y_data|)
```

## 📖 Documentation

- **MODEL_EXPLANATION.md**: Detailed explanation of the model, parameters, and coordinate transformations
- **COMPLETE_PROCESS_EXPLANATION.md**: Comprehensive step-by-step process with rationale for each decision
- **EQUATIONS.txt**: Quick reference for equations

## 🧪 Example Output

```
Initial guess:  theta=25.123456 deg, X=45.678901, M=0.012345
Refined fit:    theta=25.234567 deg, X=45.789012, M=0.012456, L1-normal-misfit=0.123456

L1 DISTANCE METRIC (Uniform Sampling)
======================================================================
Number of uniformly sampled points: 1000
Mean L1 distance:        0.234567
Median L1 distance:      0.198765
Total L1 distance:       234.567890
Max L1 distance:         1.234567
Std L1 distance:         0.156789
======================================================================
```

## 🎯 Key Features

- ✅ Robust L1-based loss functions
- ✅ Two-stage optimization (analytical + grid search)
- ✅ Adaptive parameter refinement
- ✅ Comprehensive metrics and visualization
- ✅ LaTeX/Desmos export format

## 🔍 Model Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| θ (theta) | 0.1° - 49.9° | Rotation angle of main direction |
| X | 0.0 - 100.0 | Horizontal shift parameter |
| M | -0.05 - 0.05 | Exponential modulation (amplitude) |

## 📝 Notes

- The model assumes fixed oscillation frequency (0.3 radians/unit) and vertical offset (42)
- L1 loss is used for robustness to outliers
- Grid search ensures finding optimum in search region
- Uniform sampling for L1 metric provides consistent evaluation

## 🐛 Troubleshooting

**Issue**: FileNotFoundError for xy_data.csv
- **Solution**: Ensure `xy_data.csv` is in the same directory as the script

**Issue**: Poor fit quality
- **Solution**: Check data format, verify parameter ranges are appropriate, try adjusting initial estimates

**Issue**: Slow execution
- **Solution**: Reduce grid search resolution or use fewer uniform samples for L1 metric

## 📄 License

This project is provided as-is for educational and research purposes.

## 👤 Author

Created for parametric curve fitting challenge/assignment.

## 🙏 Acknowledgments

- NumPy and Pandas for data manipulation
- Matplotlib for visualization
- Standard linear algebra libraries for optimization

---

**Note**: Even if the fit is not perfect, this implementation demonstrates a systematic approach to parametric curve fitting with clear explanations of the methodology and design decisions.

