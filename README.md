[IntegerVARFIMA_README.md](https://github.com/user-attachments/files/25359464/IntegerVARFIMA_README.md)
# Integer-Valued VARFIMA: Generation and Estimation

**Companion code repository for:**
> *Integer-Valued VARFIMA: Generation and Estimation*
> Under review at *Journal of the American Statistical Association: Theory and Methods*

This repository releases two Python scripts demonstrating core theoretical contributions of the paper: the propagation of long-memory dependence structures through mixed-type multivariate processes, and a state-space iterative optimization strategy that resolves the non-identifiable parameter surface arising from the simultaneous estimation of fractional integration and vector moving-average components in VARFIMA models.

---

## Method Overview

### 1. `LongMemoryPropagation.py` — Cross-Covariance Structure of Mixed-Type VARFIMA

This script establishes and empirically validates the theoretical cross-covariance functions of a **trivariate mixed-type VARFIMA** process, where one marginal is Gaussian and two are Poisson (integer-valued), all driven by a shared latent long-memory structure.

#### Model

Let $\mathbf{Z}_t = (Z_{1,t}, Z_{2,t}, Z_{3,t})^\top$ be a trivariate VARFIMA process with marginal fractional integration orders $d_1, d_2, d_3$ and coupling matrix $\mathbf{C}$:

$$Z_{i,t} = \xi_{i,t} + \sum_{j \neq i} C_{ij} Z_{j,t-1}$$

where each $\xi_{i,t}$ is a univariate $\text{ARFIMA}(p_i, d_i, q_i)$ innovation series generated via truncated MA representation:

$$(1 - B)^{-d_i} \varepsilon_{i,t} = \sum_{k=0}^{\infty} \psi_k^{(i)} \varepsilon_{i,t-k}, \quad \psi_0 = 1,\; \psi_k = \psi_{k-1} \cdot \frac{k-1+d_i}{k}$$

The observed trivariate process $\mathbf{Y}_t$ is then:

$$Y_{1,t} = Z_{1,t}, \quad Y_{2,t} \mid Z_{2,t} \sim \text{Poisson}(e^{Z_{2,t}}), \quad Y_{3,t} \mid Z_{3,t} \sim \text{Poisson}(e^{Z_{3,t}})$$

#### Key Theoretical Results Verified

**Theorem 1 (Poisson-Poisson cross-covariance).** For $h \in \mathbb{Z}$:

$$\gamma_{2,3}^{Y}(h) = \mathbb{E}[e^{Z_{2,t}}] \cdot \mathbb{E}[e^{Z_{3,t}}] \cdot \left( e^{\gamma_{2,3}^{Z}(h)} - 1 \right)$$

This establishes that the integer-valued pair $(Y_2, Y_3)$ inherits the long-memory decay rate of the latent $(Z_2, Z_3)$ cross-covariance, scaled nonlinearly through the moment generating function of the Gaussian process.

**Theorem 2 (Gaussian-Poisson cross-covariance).** For $h \in \mathbb{Z}$:

$$\gamma_{1,2}^{Y}(h) = \mathbb{E}[e^{Z_{2,t}}] \cdot \gamma_{1,2}^{Z}(h)$$

This shows that the Gaussian-Poisson pair shares the *same lag structure* as the latent cross-covariance, differing only by a constant scale factor $\mathbb{E}[e^{Z_2}]$.

Both theorems are verified via Monte Carlo simulation ($n_{\text{sim}} = 100$, $n = 5000$), with 95% confidence ribbons plotted over the averaged empirical cross-covariance functions.

#### Outputs

- `PP_Varfima.pdf` — Poisson-Poisson cross-covariance: $\gamma_{2,3}^{Y}(h)$ vs $\gamma_{2,3}^{\exp(Z)}(h)$ vs $\gamma_{2,3}^{Z}(h)$ vs theoretical $c \cdot \gamma_{2,3}^{Z}(h)$
- `GP_Varfima.pdf` — Gaussian-Poisson cross-covariance: $\gamma_{1,2}^{Y}(h)$ vs $\gamma_{1,2}^{\exp(Z)}(h)$ vs $\gamma_{1,2}^{Z}(h)$ vs theoretical $c \cdot \gamma_{1,2}^{Z}(h)$
- `GPP_Varfima.pdf` — Sample time series plots for $Z_1, Z_2, Z_3$, $Y_1, Y_2, Y_3$, $\lambda_2 = e^{Z_2}$, $\lambda_3 = e^{Z_3}$ over the first 500 observations

---

### 2. `StateSpaceVARFIMA.py` — State-Space Iterative Estimation of VARFIMA(1, D, 1)

This script addresses the central estimation challenge of the paper: **the log-likelihood surface of a VARFIMA(1, D, 1) model is not separable in the fractional integration order $\mathbf{d}$ and the VMA coefficient $\boldsymbol{\Theta}$**, making joint optimization over the full parameter space highly ill-conditioned.

#### The Identification Problem

A $k$-variate VARFIMA$(1, \mathbf{D}, 1)$ model takes the form:

$$\boldsymbol{\Phi}(B)(1-B)^{\mathbf{D}} \mathbf{X}_t = \boldsymbol{\Theta}(B) \boldsymbol{\varepsilon}_t, \quad \boldsymbol{\varepsilon}_t \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})$$

where $\mathbf{D} = \text{diag}(d_1, \ldots, d_k)$ is the fractional differencing matrix. After fractional differencing $\mathbf{Y}_t = (1-B)^{\mathbf{D}} \mathbf{X}_t$, the model reduces to a $\text{VAR}(1)$-$\text{VMA}(1)$ system:

$$\mathbf{Y}_t = \boldsymbol{\Phi} \mathbf{Y}_{t-1} + \boldsymbol{\varepsilon}_t + \boldsymbol{\Theta} \boldsymbol{\varepsilon}_{t-1}$$

The mixed VMA(1)–FI structure creates **two inseparable sources of autocorrelation**: the VMA(1) term $\boldsymbol{\Theta}\boldsymbol{\varepsilon}_{t-1}$ introduces short-range dependence that partially mimics fractional integration, and vice versa. The joint log-likelihood:

$$\ell(\mathbf{d}, \boldsymbol{\Phi}, \boldsymbol{\Theta}, \boldsymbol{\Sigma}) = -\frac{n-1}{2} \log |\boldsymbol{\Sigma}| - \frac{1}{2} \sum_{t=2}^{n} \boldsymbol{\varepsilon}_t^\top \boldsymbol{\Sigma}^{-1} \boldsymbol{\varepsilon}_t$$

where the innovations $\boldsymbol{\varepsilon}_t = \mathbf{Y}_t - \boldsymbol{\Phi}\mathbf{Y}_{t-1} - \boldsymbol{\Theta}\boldsymbol{\varepsilon}_{t-1}$ depend on $\mathbf{d}$ through $\mathbf{Y}_t$, is **non-convex and riddled with saddle points** when $\mathbf{d}$ and $\boldsymbol{\Theta}$ are optimized jointly. Standard gradient-based optimizers routinely converge to degenerate solutions where $\hat{d}_j \approx 0$ and $\hat{\boldsymbol{\Theta}}$ absorbs all memory structure, or vice versa.

#### The State-Space Iterative Solution

The key contribution of `StateSpaceVARFIMA.py` is decomposing this non-separable surface by treating the fractionally differenced series as a **latent state**, and optimizing over a reduced parameter space at each step:

**Step 1 — Fractional differencing (state extraction).**
Given current $\mathbf{d}^{(r)}$, apply the truncated fractional differencing filter using the $\pi$-weight recursion:

$$\pi_0 = 1, \quad \pi_k = \pi_{k-1} \cdot \frac{k - 1 - d_j}{k}, \quad \mathbf{Y}_t^{(r)} = \sum_{k=0}^{t} \pi_k \mathbf{X}_{t-k}$$

This maps $\mathbf{X}_t$ to the latent VARMA residuals $\mathbf{Y}_t^{(r)}$ conditioned on the current memory estimate.

**Step 2 — VARMA likelihood (conditional optimization).**
Given $\mathbf{Y}_t^{(r)}$, optimize $(\boldsymbol{\Phi}, \boldsymbol{\Theta}, \boldsymbol{\Sigma})$ over a well-conditioned surface where the short-memory structure is no longer confounded with fractional integration. The innovations are computed via the recursive filter:

$$\boldsymbol{\varepsilon}_t^{(r)} = \mathbf{Y}_t^{(r)} - \boldsymbol{\Phi} \mathbf{Y}_{t-1}^{(r)} - \boldsymbol{\Theta} \boldsymbol{\varepsilon}_{t-1}^{(r)}$$

**Step 3 — Memory update.**
Update $\mathbf{d}^{(r+1)}$ by minimizing the full log-likelihood over $\mathbf{d}$ alone, with $(\boldsymbol{\Phi}, \boldsymbol{\Theta}, \boldsymbol{\Sigma})$ held fixed from Step 2.

This profile-likelihood iteration — alternating between state-space extraction and conditional VARMA optimization — avoids the degenerate saddle regions of the joint surface. The Cholesky parameterization $\boldsymbol{\Sigma} = \mathbf{L}\mathbf{L}^\top$ ensures positive definiteness throughout, and L-BFGS-B is applied with hard bounds $|d_j| < 0.49$, $\|\boldsymbol{\Phi}\|_{\text{spec}} < 0.99$ to enforce stationarity.

#### Monte Carlo Validation

A Monte Carlo study ($n_{\text{sim}} = 50$, $n = 5000$, $k = 2$) evaluates finite-sample estimator performance across all parameters $(\mathbf{d}, \boldsymbol{\Phi}, \boldsymbol{\Theta})$ under true values:

$$d = (0.25, 0.35), \quad \boldsymbol{\Phi} = \begin{pmatrix} 0.2 & 0.1 \\ 0.05 & 0.3 \end{pmatrix}, \quad \boldsymbol{\Theta} = \begin{pmatrix} 0.15 & 0.05 \\ 0.1 & 0.2 \end{pmatrix}, \quad \boldsymbol{\Sigma} = \begin{pmatrix} 1.0 & 0.3 \\ 0.3 & 1.0 \end{pmatrix}$$

#### Output

- `varfima_estimation_histograms.pdf` — Boxplot panel for all estimated parameters across Monte Carlo runs (box: 25/75th percentile, whiskers: 2.5/97.5th percentile, dot: mean, cross: true value)

---

## File Structure

```
IntegerVARFIMA/
│
├── LongMemoryPropagation.py     # Trivariate mixed-type VARFIMA simulation
│                                # + cross-covariance validation (Theorems 1 & 2)
│
├── StateSpaceVARFIMA.py         # State-space iterative VARFIMA(1,D,1) estimation
│                                # + Monte Carlo parameter recovery study
│
├── outputs/
│   ├── PP_Varfima.pdf           # Poisson-Poisson cross-covariance plot
│   ├── GP_Varfima.pdf           # Gaussian-Poisson cross-covariance plot
│   ├── GPP_Varfima.pdf          # Sample time series (first 500 obs)
│   └── varfima_estimation_histograms.pdf   # Monte Carlo boxplots
│
└── README.md
```

---

## Dependencies

```bash
pip install numpy scipy matplotlib
```

Tested on **Python ≥ 3.9**.

---

## How to Run

### `LongMemoryPropagation.py`

```bash
python LongMemoryPropagation.py
```

Set the output directory and model parameters at the top of the script:

```python
out_dir  = "path/to/outputs"   # PDF output directory
n_sims   = 100                 # Number of Monte Carlo replications
n        = 5000                # Series length per replication
max_lag  = 50                  # Maximum lag for cross-covariance
```

Expected runtime: ~5–15 minutes for `n_sims = 100`, `n = 5000`, depending on hardware.

### `StateSpaceVARFIMA.py`

```bash
python StateSpaceVARFIMA.py
```

Configure the Monte Carlo study at the bottom of the script:

```python
results = monte_carlo_varfima_1d1(
    n_sim = 50,       # Number of replications
    n     = 5000,     # Series length
    true_params = dict(
        d     = np.array([0.25, 0.35]),
        Phi   = np.array([[0.2, 0.1], [0.05, 0.3]]),
        Theta = np.array([[0.15, 0.05], [0.1, 0.2]]),
        Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
    )
)
```

The histogram PDF is saved to the working directory by default. Pass a full path to `plot_histograms(..., pdf_path="path/to/varfima_estimation_histograms.pdf")` to redirect it.

Expected runtime: ~20–60 minutes for `n_sim = 50`, `n = 5000`, depending on optimizer convergence.

---

## Notes

- **Reproducibility:** Both scripts fix `np.random.seed(123)` (`LongMemoryPropagation.py`) and per-simulation seeds via `base_seed + sim` (`StateSpaceVARFIMA.py`) for full reproducibility.
- **Convergence:** `StateSpaceVARFIMA.py` silently skips replications where L-BFGS-B does not converge (`convergence = False`). The Monte Carlo summary statistics are computed over converged runs only; the number of successful replications is implicitly reflected in the boxplot sample sizes.
- **Truncation:** The fractional MA representation is truncated at the full series length. For large $n$, this is computationally exact to floating-point precision but scales as $O(n^2)$ per series. For exploratory runs, reduce `n` to 1000.
- **Partial release:** This repository covers the simulation and estimation components. Additional code for theoretical derivations, empirical applications, and supplementary figures will be released upon acceptance.
