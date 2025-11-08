"""
Parametric Curve Fitting Algorithm

This script fits a 2D parametric curve to data points using a rotated coordinate
system with exponentially modulated sinusoidal oscillations.

Model: x(t) = t·cos(θ) - e^(M|t|)·sin(0.3t)·sin(θ) + X
       y(t) = 42 + t·sin(θ) + e^(M|t|)·sin(0.3t)·cos(θ)

Parameters: θ (rotation angle), X (horizontal shift), M (exponential modulation)
"""

import numpy as np, pandas as pd, math

import matplotlib.pyplot as plt



# --- Load data ---

df = pd.read_csv("xy_data.csv")   # columns must be x,y (case-insensitive)

x = df[df.columns[0]].to_numpy().astype(float)

y = df[df.columns[1]].to_numpy().astype(float)



# --- Helpers ---

def deg2rad(d): return d * math.pi / 180.0



def rotate_and_project(x, y, theta_deg, X):

    th = deg2rad(theta_deg)

    ct, st = math.cos(th), math.sin(th)

    # Shift by (X, 42)

    Xc = x - X

    Yc = y - 42.0

    # Projections along (t-direction) and normal:

    t =  Xc*ct + Yc*st                    # should align with t

    v = -Xc*st + Yc*ct                    # should be e^{M|t|} sin(0.3 t)

    return t, v



def fit_theta_linear(x, y):

    # slope from least squares y = a + b x

    X = np.vstack([np.ones_like(x), x]).T

    b = np.linalg.lstsq(X, y, rcond=None)[0][1]

    theta = np.degrees(np.arctan2(b, 1.0))

    # Clamp to (0,50)

    return float(np.clip(theta, 0.1, 49.9))



def choose_X_for_t_range(x, y, theta, target_mean=33.0):

    th = deg2rad(theta)

    ct, st = math.cos(th), math.sin(th)

    # t(X=0) = x*ct + (y-42)*st

    t0 = x*ct + (y-42.0)*st

    mean_t0 = float(np.median(t0))

    # t shifts by -X*ct, so set mean_t = target_mean => X = (mean_t0 - target_mean)/ct

    X = (mean_t0 - target_mean) / ct

    return float(np.clip(X, 0.0, 100.0))



def fit_M(t, v):

    # Use points with |sin(0.3 t)| not too small

    s = np.sin(0.3 * t)

    mask = np.abs(s) > 0.2

    if np.count_nonzero(mask) < 5:

        return 0.0

    z = np.log(np.clip(np.abs(v[mask] / s[mask]), 1e-12, None))

    w = np.abs(t[mask])

    # Linear least squares without intercept: z ≈ M * |t|

    denom = np.sum(w*w)

    if denom < 1e-9:

        return 0.0

    M = float(np.sum(w * z) / denom)

    # Clamp to (-0.05, 0.05)

    return float(np.clip(M, -0.05, 0.05))



def l1_normal_misfit(theta, X, M, x, y):

    t, v = rotate_and_project(x, y, theta, X)

    pred = np.exp(np.abs(t)*M) * np.sin(0.3*t)

    return float(np.mean(np.abs(v - pred)))



# --- Initial estimates ---

theta0 = fit_theta_linear(x, y)

X0     = choose_X_for_t_range(x, y, theta0, target_mean=33.0)

t0, v0 = rotate_and_project(x, y, theta0, X0)

M0     = fit_M(t0, v0)



# --- Local refinement: small grid around (theta0, X0, M0) minimizing L1 normal misfit ---

def refine(theta0, X0, M0, x, y):

    theta_grid = np.linspace(max(0.1, theta0-3.0), min(49.9, theta0+3.0), 31)

    X_grid     = np.linspace(max(0.0, X0-5.0),      min(100.0, X0+5.0),   41)

    M_grid     = np.linspace(max(-0.05, M0-0.01),   min(0.05,  M0+0.01),  41)



    best = (1e9, theta0, X0, M0)

    # Coarse sweep over theta + X, re-fit M at each pair, then fine tune M

    for th in theta_grid:

        for Xv in X_grid:

            t, v = rotate_and_project(x, y, th, Xv)

            M_guess = fit_M(t, v)

            # Small band around M_guess

            m_lo = max(-0.05, M_guess - 0.01)

            m_hi = min( 0.05, M_guess + 0.01)

            for Mv in np.linspace(m_lo, m_hi, 21):

                mis = l1_normal_misfit(th, Xv, Mv, x, y)

                if mis < best[0]:

                    best = (mis, th, Xv, Mv)

    return {"misfit": best[0], "theta": best[1], "X": best[2], "M": best[3]}



res = refine(theta0, X0, M0, x, y)

theta, X, M = res["theta"], res["X"], res["M"]



print("Initial guess:  theta={:.6f} deg, X={:.6f}, M={:.6f}".format(theta0, X0, M0))

print("Refined fit:    theta={:.6f} deg, X={:.6f}, M={:.6f}, L1-normal-misfit={:.6g}".format(theta, X, M, res["misfit"]))



# --- Build Desmos/LaTeX submission string ---

latex = (

    "\\left(t*\\cos({th})-e^{{{M}\\left|t\\right|}}\\cdot\\sin(0.3t)\\sin({th})+{X},"

    "42+t*\\sin({th})+e^{{{M}\\left|t\\right|}}\\cdot\\sin(0.3t)\\cos({th})\\right)"

).format(th=np.deg2rad(theta), M=M, X=X)

print("\nDesmos/LaTeX submission:\n", latex)



# --- Visual check: sample model and overlay data ---

tt = np.linspace(6.0, 60.0, 600)

th = np.deg2rad(theta)

ct, st = np.cos(th), np.sin(th)

x_model =  tt*ct - np.exp(np.abs(tt)*M)*np.sin(0.3*tt)*st + X

y_model = 42 + tt*st + np.exp(np.abs(tt)*M)*np.sin(0.3*tt)*ct



plt.figure(figsize=(6,6))

plt.plot(x, y, ".", label="data")

plt.plot(x_model, y_model, "-", label="fit")

plt.axis("equal")

plt.legend()

plt.title("Data vs Fitted Parametric Curve")

plt.xlabel("x"); plt.ylabel("y")

plt.show()


# --- Calculate L1 distance between uniformly sampled points ---

def calculate_l1_distance_uniform_samples(theta, X, M, x_data, y_data, n_samples=1000, t_range=None):
    """
    Calculate L1 (Manhattan) distance between uniformly sampled points on predicted curve
    and nearest actual data points.
    
    Parameters:
    -----------
    theta : float
        Rotation angle in degrees
    X : float
        Horizontal shift
    M : float
        Exponential modulation parameter
    x_data : array
        Actual x coordinates
    y_data : array
        Actual y coordinates
    n_samples : int
        Number of uniformly sampled points
    t_range : tuple or None
        (t_min, t_max) range. If None, determines from data.
    
    Returns:
    --------
    dict with:
        - l1_distance: Mean L1 distance per sample point
        - total_l1_distance: Total L1 distance
        - max_l1_distance: Maximum L1 distance
        - median_l1_distance: Median L1 distance
    """
    th = np.deg2rad(theta)
    ct, st = np.cos(th), np.sin(th)
    
    # Determine t range from data if not provided
    if t_range is None:
        t_data, _ = rotate_and_project(x_data, y_data, theta, X)
        t_min, t_max = float(np.min(t_data)), float(np.max(t_data))
        # Add small buffer
        t_range = (t_min - 1.0, t_max + 1.0)
    
    t_min, t_max = t_range
    
    # Uniformly sample t values
    t_samples = np.linspace(t_min, t_max, n_samples)
    
    # Generate predicted points from model
    x_pred = t_samples*ct - np.exp(np.abs(t_samples)*M)*np.sin(0.3*t_samples)*st + X
    y_pred = 42 + t_samples*st + np.exp(np.abs(t_samples)*M)*np.sin(0.3*t_samples)*ct
    
    # Stack predicted points
    pred_points = np.column_stack([x_pred, y_pred])
    data_points = np.column_stack([x_data, y_data])
    
    # Calculate L1 distance to nearest neighbor for each predicted point
    # Vectorized computation for efficiency
    l1_distances = np.array([
        np.min(np.abs(pred_pt[0] - data_points[:, 0]) + np.abs(pred_pt[1] - data_points[:, 1]))
        for pred_pt in pred_points
    ])
    
    return {
        "mean_l1_distance": float(np.mean(l1_distances)),
        "total_l1_distance": float(np.sum(l1_distances)),
        "max_l1_distance": float(np.max(l1_distances)),
        "median_l1_distance": float(np.median(l1_distances)),
        "std_l1_distance": float(np.std(l1_distances)),
        "n_samples": n_samples
    }


# Calculate L1 distance metric
print("\n" + "="*70)
print("L1 DISTANCE METRIC (Uniform Sampling)")
print("="*70)
l1_results = calculate_l1_distance_uniform_samples(theta, X, M, x, y, n_samples=1000)
print(f"Number of uniformly sampled points: {l1_results['n_samples']}")
print(f"Mean L1 distance:        {l1_results['mean_l1_distance']:.6f}")
print(f"Median L1 distance:      {l1_results['median_l1_distance']:.6f}")
print(f"Total L1 distance:       {l1_results['total_l1_distance']:.6f}")
print(f"Max L1 distance:         {l1_results['max_l1_distance']:.6f}")
print(f"Std L1 distance:         {l1_results['std_l1_distance']:.6f}")
print("="*70)

