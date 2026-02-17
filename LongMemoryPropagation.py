import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── ARFIMA simulation ──────────────────────────────────────────────────────────

def generate_arfima(phi, d, theta, sigma, n, burnin=1000):
    """Generate a univariate ARFIMA(p,d,q) series via truncated MA representation."""
    total = n + burnin
    innovations = np.random.normal(0, sigma, total)

    # Fractional integration coefficients psi_k: (1-B)^{-d} = sum psi_k B^k
    # psi_0 = 1, psi_k = psi_{k-1} * (k-1+d) / k
    psi = np.ones(total)
    for k in range(1, total):
        psi[k] = psi[k - 1] * (k - 1 + d) / k

    # Fractionally integrated innovations
    fd = np.zeros(total)
    for t in range(total):
        fd[t] = np.dot(psi[:t + 1], innovations[t::-1])

    # AR coefficients
    ar = np.array([phi] if np.isscalar(phi) else phi, dtype=float)
    ar = ar[ar != 0]
    p = len(ar)

    # MA coefficients
    ma = np.array([theta] if np.isscalar(theta) else theta, dtype=float)
    ma = ma[ma != 0]
    q = len(ma)

    # Apply ARMA filter on top
    series = np.zeros(total)
    for t in range(total):
        val = fd[t]
        for i in range(min(t, p)):
            val += ar[i] * series[t - 1 - i]
        for j in range(min(t, q)):
            val += ma[j] * innovations[t - 1 - j]
        series[t] = val

    return series[burnin:]


# ── Trivariate mixed VARFIMA ───────────────────────────────────────────────────

def generate_trivariate_varfima(
    phi1, d1, theta1, sigma1,
    phi2, d2, theta2, sigma2,
    phi3, d3, theta3, sigma3,
    coupling_matrix, n, burnin=1000
):
    arfima1 = generate_arfima(phi1, d1, theta1, sigma1, n, burnin)
    arfima2 = generate_arfima(phi2, d2, theta2, sigma2, n, burnin)
    arfima3 = generate_arfima(phi3, d3, theta3, sigma3, n, burnin)

    Z1 = np.zeros(n)
    Z2 = np.zeros(n)
    Z3 = np.zeros(n)

    Z1[0] = arfima1[0]
    Z2[0] = arfima2[0]
    Z3[0] = arfima3[0]

    for t in range(1, n):
        Z1[t] = arfima1[t] + coupling_matrix[0, 1] * Z2[t-1] + coupling_matrix[0, 2] * Z3[t-1]
        Z2[t] = arfima2[t] + coupling_matrix[1, 0] * Z1[t-1] + coupling_matrix[1, 2] * Z3[t-1]
        Z3[t] = arfima3[t] + coupling_matrix[2, 0] * Z1[t-1] + coupling_matrix[2, 1] * Z2[t-1]

    Y1      = Z1.copy()
    lambda2 = np.exp(Z2)
    lambda3 = np.exp(Z3)
    Y2      = np.random.poisson(lambda2)
    Y3      = np.random.poisson(lambda3)

    return dict(Z1=Z1, Z2=Z2, Z3=Z3,
                Y1=Y1, Y2=Y2, Y3=Y3,
                lambda2=lambda2, lambda3=lambda3)


# ── Cross-covariance ───────────────────────────────────────────────────────────

def cross_cov(x, y, max_lag):
    n = len(x)
    mx = np.mean(x)
    my = np.mean(y)
    cov_values = np.zeros(2 * max_lag + 1)

    for h in range(max_lag, 0, -1):               # negative lags
        idx = max_lag - h
        cov_values[idx] = np.sum((x[h:] - mx) * (y[:n-h] - my)) / (n - h)

    cov_values[max_lag] = np.sum((x - mx) * (y - my)) / n   # lag 0

    for h in range(1, max_lag + 1):               # positive lags
        idx = max_lag + h
        cov_values[idx] = np.sum((x[:n-h] - mx) * (y[h:] - my)) / (n - h)

    return cov_values


# ── Single simulation run ──────────────────────────────────────────────────────

def run_simulation(sim_number, phi1, d1, theta1, sigma1,
                   phi2, d2, theta2, sigma2,
                   phi3, d3, theta3, sigma3,
                   coupling, n, max_lag):

    result = generate_trivariate_varfima(
        phi1, d1, theta1, sigma1,
        phi2, d2, theta2, sigma2,
        phi3, d3, theta3, sigma3,
        coupling, n, burnin=1000
    )

    covs = dict(
        cov_z1_z2         = cross_cov(result["Z1"],      result["Z2"],      max_lag),
        cov_z1_z3         = cross_cov(result["Z1"],      result["Z3"],      max_lag),
        cov_z2_z3         = cross_cov(result["Z2"],      result["Z3"],      max_lag),
        cov_y1_y2         = cross_cov(result["Y1"],      result["Y2"],      max_lag),
        cov_y1_y3         = cross_cov(result["Y1"],      result["Y3"],      max_lag),
        cov_y2_y3         = cross_cov(result["Y2"],      result["Y3"],      max_lag),
        cov_exp_z2_exp_z3 = cross_cov(result["lambda2"], result["lambda3"], max_lag),
        cov_z1_exp_z2     = cross_cov(result["Z1"],      result["lambda2"], max_lag),
        cov_y1_z2         = cross_cov(result["Y1"],      result["Z2"],      max_lag),
        cov_y1_z3         = cross_cov(result["Y1"],      result["Z3"],      max_lag),
        cov_y2_z1         = cross_cov(result["Y2"],      result["Z1"],      max_lag),
        cov_y2_z3         = cross_cov(result["Y2"],      result["Z3"],      max_lag),
        cov_y3_z1         = cross_cov(result["Y3"],      result["Z1"],      max_lag),
        cov_y3_z2         = cross_cov(result["Y3"],      result["Z2"],      max_lag),
    )

    mean_exp_z2 = np.mean(result["lambda2"])
    mean_exp_z3 = np.mean(result["lambda3"])
    covs["scale_factor_z2z3"] = mean_exp_z2 * mean_exp_z3 * (np.exp(covs["cov_z2_z3"]) - 1)
    covs["scale_factor_z1z2"] = mean_exp_z2

    if sim_number == 1:
        covs["_series"] = result   # keep series from run 1 for time series plots

    return covs


# ── Parameters ─────────────────────────────────────────────────────────────────

np.random.seed(123)

phi1   = 0.3;  d1 = 0.3;  theta1 = 0.2;  sigma1 = 1.0
phi2   = 0.4;  d2 = 0.2;  theta2 = 0.1;  sigma2 = 0.5
phi3   = 0.25; d3 = 0.35; theta3 = 0.15; sigma3 = 0.7

coupling = np.array([[1.0, 0.2, 0.1],
                     [0.3, 1.0, 0.2],
                     [0.1, 0.3, 1.0]])

n       = 5000
max_lag = 50
n_sims  = 100

# ── Run simulations ────────────────────────────────────────────────────────────

print(f"Running {n_sims} simulations...")
sim_results = []
for i in range(1, n_sims + 1):
    if i % 10 == 0:
        print(i, end=" ", flush=True)
    sim_results.append(
        run_simulation(i, phi1, d1, theta1, sigma1,
                       phi2, d2, theta2, sigma2,
                       phi3, d3, theta3, sigma3,
                       coupling, n, max_lag)
    )
print()

# ── Aggregate results ──────────────────────────────────────────────────────────

cov_types = [
    "cov_z1_z2", "cov_z1_z3", "cov_z2_z3",
    "cov_y1_y2", "cov_y1_y3", "cov_y2_y3",
    "cov_exp_z2_exp_z3", "cov_z1_exp_z2",
    "cov_y1_z2", "cov_y1_z3",
    "cov_y2_z1", "cov_y2_z3",
    "cov_y3_z1", "cov_y3_z2"
]

avg_covs = {t: np.mean([r[t] for r in sim_results], axis=0) for t in cov_types}

avg_scale_z2z3 = np.mean([r["scale_factor_z2z3"] for r in sim_results], axis=0)
avg_scale_z1z2 = np.mean([r["scale_factor_z1z2"] for r in sim_results])

theoretical_exp_z2_exp_z3 = avg_scale_z2z3
theoretical_z1_exp_z2     = avg_covs["cov_z1_z2"] * avg_scale_z1z2

key_pairs = ["cov_z2_z3", "cov_y2_y3", "cov_exp_z2_exp_z3",
             "cov_z1_z2", "cov_y1_y2", "cov_z1_exp_z2"]
sd_values = {t: np.std([r[t] for r in sim_results], axis=0, ddof=1) for t in key_pairs}

lags = np.arange(-max_lag, max_lag + 1)

first_series = sim_results[0]["_series"]

# ── Plot settings ──────────────────────────────────────────────────────────────

LEGEND_FS    = 28
TITLE_FS     = 28
AXLABEL_FS   = 24
AXTICK_FS    = 20
LINE_LW      = 2.2
RIBBON_ALPHA = 0.20
out_dir      = "D:/Astro2"


def add_ribbon(ax, x, y, sd, n_s, color):
    lower = y - 1.96 * sd / np.sqrt(n_s)
    upper = y + 1.96 * sd / np.sqrt(n_s)
    ax.fill_between(x, lower, upper, color=color, alpha=RIBBON_ALPHA, linewidth=0)


def save_fig(fig, path):
    fig.savefig(path, format="pdf", bbox_inches="tight")
    print(f"Saved: {path}")


# ── Plot 1: Poisson-Poisson covariance ────────────────────────────────────────

fig1, ax1 = plt.subplots(figsize=(20, 14))

series_pp = [
    (avg_covs["cov_y2_y3"],          sd_values["cov_y2_y3"],          "solid",  "blue",   r"$\gamma_{2,3}^{Y}(h)$"),
    (avg_covs["cov_exp_z2_exp_z3"],  sd_values["cov_exp_z2_exp_z3"],  "solid",  "orange", r"$\gamma_{2,3}^{\exp(Z)}(h)$"),
    (avg_covs["cov_z2_z3"],          sd_values["cov_z2_z3"],          "solid",  "green",  r"$\gamma_{2,3}^{Z}(h)$"),
    (theoretical_exp_z2_exp_z3,      None,                            "dashed", "red",    r"$c \cdot \gamma_{2,3}^{Z}(h)$"),
]

for data, sd, ls, col, lbl in series_pp:
    ax1.plot(lags, data, color=col, linestyle=ls, linewidth=LINE_LW, label=lbl)
    if sd is not None:
        add_ribbon(ax1, lags, data, sd, n_sims, col)

ax1.set_title("Covariance Functions of Poisson-Poisson Pairs",
              fontsize=TITLE_FS, fontweight="bold")
ax1.set_xlabel("Lag (h)", fontsize=AXLABEL_FS)
ax1.set_ylabel("", fontsize=AXLABEL_FS)
ax1.tick_params(labelsize=AXTICK_FS)

legend_handles = [
    Line2D([0], [0], color="blue",   linestyle="solid",  linewidth=LINE_LW, label=r"$\gamma_{2,3}^{Y}(h)$"),
    Line2D([0], [0], color="orange", linestyle="solid",  linewidth=LINE_LW, label=r"$\gamma_{2,3}^{\exp(Z)}(h)$"),
    Line2D([0], [0], color="green",  linestyle="solid",  linewidth=LINE_LW, label=r"$\gamma_{2,3}^{Z}(h)$"),
    Line2D([0], [0], color="red",    linestyle="dashed", linewidth=LINE_LW, label=r"$c \cdot \gamma_{2,3}^{Z}(h)$"),
]
ax1.legend(handles=legend_handles, fontsize=LEGEND_FS, loc="upper right",
           framealpha=1.0, handlelength=5, handleheight=1.8,
           borderpad=1.0, labelspacing=0.8, handletextpad=0.8)

plt.tight_layout()
save_fig(fig1, f"{out_dir}/PP_Varfima.pdf")
plt.close(fig1)

# ── Plot 2: Gaussian-Poisson covariance ───────────────────────────────────────

fig2, ax2 = plt.subplots(figsize=(20, 14))

series_gp = [
    (avg_covs["cov_y1_y2"],     sd_values["cov_y1_y2"],     "solid",  "blue",   r"$\gamma_{1,2}^{Y}(h)$"),
    (avg_covs["cov_z1_exp_z2"], sd_values["cov_z1_exp_z2"], "solid",  "orange", r"$\gamma_{1,2}^{\exp(Z)}(h)$"),
    (avg_covs["cov_z1_z2"],     sd_values["cov_z1_z2"],     "solid",  "green",  r"$\gamma_{1,2}^{Z}(h)$"),
    (theoretical_z1_exp_z2,     None,                       "dashed", "red",    r"$c \cdot \gamma_{1,2}^{Z}(h)$"),
]

for data, sd, ls, col, lbl in series_gp:
    ax2.plot(lags, data, color=col, linestyle=ls, linewidth=LINE_LW, label=lbl)
    if sd is not None:
        add_ribbon(ax2, lags, data, sd, n_sims, col)

ax2.set_title("Covariance Functions of Gaussian-Poisson Pairs",
              fontsize=TITLE_FS, fontweight="bold")
ax2.set_xlabel("Lag (h)", fontsize=AXLABEL_FS)
ax2.set_ylabel("", fontsize=AXLABEL_FS)
ax2.tick_params(labelsize=AXTICK_FS)

legend_handles2 = [
    Line2D([0], [0], color="blue",   linestyle="solid",  linewidth=LINE_LW, label=r"$\gamma_{1,2}^{Y}(h)$"),
    Line2D([0], [0], color="orange", linestyle="solid",  linewidth=LINE_LW, label=r"$\gamma_{1,2}^{\exp(Z)}(h)$"),
    Line2D([0], [0], color="green",  linestyle="solid",  linewidth=LINE_LW, label=r"$\gamma_{1,2}^{Z}(h)$"),
    Line2D([0], [0], color="red",    linestyle="dashed", linewidth=LINE_LW, label=r"$c \cdot \gamma_{1,2}^{Z}(h)$"),
]
ax2.legend(handles=legend_handles2, fontsize=LEGEND_FS, loc="upper right",
           framealpha=1.0, handlelength=5, handleheight=1.8,
           borderpad=1.0, labelspacing=0.8, handletextpad=0.8)

plt.tight_layout()
save_fig(fig2, f"{out_dir}/GP_Varfima.pdf")
plt.close(fig2)

# ── Plot 3: Time series ────────────────────────────────────────────────────────

plot_window = 500   # show first 500 observations
t_axis = np.arange(plot_window)

ts_specs = [
    (first_series["Z1"][:plot_window],      "steelblue",  r"$Z_1$"),
    (first_series["Z2"][:plot_window],      "darkorange", r"$Z_2$"),
    (first_series["Z3"][:plot_window],      "seagreen",   r"$Z_3$"),
    (first_series["Y1"][:plot_window],      "steelblue",  r"$Y_1$ (Gaussian)"),
    (first_series["Y2"][:plot_window],      "darkorange", r"$Y_2$ (Poisson)"),
    (first_series["Y3"][:plot_window],      "seagreen",   r"$Y_3$ (Poisson)"),
    (first_series["lambda2"][:plot_window], "darkorange", r"$\exp(Z_2) = \lambda_2$"),
    (first_series["lambda3"][:plot_window], "seagreen",   r"$\exp(Z_3) = \lambda_3$"),
]

fig3, axes = plt.subplots(4, 2, figsize=(16, 16), sharex=True)
axes_flat = axes.flatten()

for ax, (data, col, label) in zip(axes_flat, ts_specs):
    ax.plot(t_axis, data, color=col, linewidth=0.9)
    ax.set_title(label, fontsize=TITLE_FS, fontweight="bold")
    ax.tick_params(labelsize=AXTICK_FS)
    ax.set_ylabel("Value", fontsize=AXLABEL_FS - 2)

for ax in axes_flat[-2:]:
    ax.set_xlabel("Time", fontsize=AXLABEL_FS)

fig3.suptitle(
    "First 500 observations of Sample Time Series",
    fontsize=TITLE_FS + 2, fontweight="bold"
)
plt.tight_layout()
save_fig(fig3, f"{out_dir}/GPP_Varfima.pdf")
plt.close(fig3)

print("All done.")
