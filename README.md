# Parametric Curve Fitting Project

Fitting a 2D parametric curve to data points using a rotated coordinate system with exponentially modulated sinusoidal oscillations.

## 📋 Overview

This project fits a curve of the form:

```
x(t) = t·cos(θ) - e^(M|t|)·sin(0.3t)·sin(θ) + X
y(t) = 42 + t·sin(θ) + e^(M|t|)·sin(0.3t)·cos(θ)
```

Where the model parameters **θ (rotation angle)**, **X (horizontal shift)**, and **M (exponential modulation)** are estimated from data.

---

## ✅ Final Fitted Equation (Solved Result)

```
\left(
 t*cos(0.5227505495584853)
 - e^{0.029972582162542474|t|}*sin(0.3t)*sin(0.5227505495584853)
 + 55.01160286897831,

 42 + t*sin(0.5227505495584853)
 + e^{0.029972582162542474|t|}*sin(0.3t)*cos(0.5227505495584853)

ight)
```

You may paste the above into **Desmos** to visualize.

---

## 🚀 Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Curve Fitting

```bash
python fit_parametric_curve.py
```

### Output (Printed to Console)

- Initial parameter estimates
- Refined parameters after grid search
- L1 misfit metrics (mean, median, max, total)
- Final equation (LaTeX/Desmos-ready)
- Plot visualization of fitted curve vs data points

---

## 📁 Project Structure

```
.
├── fit_parametric_curve.py              # Main fitting program
├── xy_data.csv                          # Input data file (x,y pairs)
├── requirements.txt                     # Python dependencies
├── README.md                            # Documentation
├── MODEL_EXPLANATION.md                 # Mathematical derivations
├── COMPLETE_PROCESS_EXPLANATION.md      # Detailed step-by-step breakdown
└── EQUATIONS.txt                        # Quick parametric equation reference
```

---

## 🔧 Algorithm Overview

### Stage 1: Analytical estimation

| Parameter | Method |
|----------|--------|
| θ (theta) | Linear regression on main direction |
| X | Center shift using t–mean |
| M | Fit exponential envelope using log-linear regression |

### Stage 2: Grid Search Refinement

- Search ±3° around θ₀ (31 points)
- Search ±5 around X₀ (41 points)
- For each (θ, X), re-fit M within ±0.01 (21 points)
- Evaluate using **L1 normal misfit**

---

## 📊 Metrics

### L1 Normal Misfit

```
misfit = mean(|v_actual - e^(M|t|)*sin(0.3t)|)
```

### L1 Distance Metric

```
L1_distance = mean(|x_pred - x_data| + |y_pred - y_data|)
```

---

## 🧪 Example Console Output

```
Initial guess: theta=25.123456°, X=45.678901, M=0.012345
Refined fit:   theta=25.234567°, X=45.789012, M=0.012456

L1 DISTANCE METRIC (Uniform Sampling)
======================================================================
Number of sample points:    1000
Mean L1 distance:           0.234567
Median L1 distance:         0.198765
Max L1 distance:            1.234567
Total L1 distance:          234.567890
======================================================================
```

---

## 🔍 Model Parameters & Constraints

| Parameter | Range |
|----------|--------|
| θ (theta) | `0.1°` – `49.9°` |
| X | `0` – `100` |
| M | `-0.05` – `0.05` |

---

## 🐛 Troubleshooting

| Issue | Solution |
|------|----------|
| `FileNotFoundError: xy_data.csv` | Ensure CSV is in project root |
| Slow execution | Reduce grid-search resolution |
| Poor fit | Verify CSV data format and sampling |

---

## 📄 License

This project is provided **as-is** for academic and research use.

---

## 👤 Author

> Designed as part of a research curve-fitting assignment.

