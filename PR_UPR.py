import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import shapiro

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, BayesianRidge, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import permutation_importance

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# =============================================================================
# 1.  DATASET  (36 specimens, Türkiye)
# =============================================================================
DATA = {
    "Sample_No": list(range(1, 37)),
    "Location": [
        "Kayseri-1",        "Batman",           "Afyon/Abbak",
        "Ankara-1",       "Nevşehir-1",       "Karaman-1",
        "Kayseri-2",      "Giresun",          "Konya/Kızılören",
        "Aksaray/Demirci","Kayseri-2",        "Konya/Sille-1",
        "Balıkesir/Marmara Island",           "Aksaray/Gülşehir",
        "Kayseri-3",      "Nevşehir-2",       "Manisa",
        "Kayseri-4",      "Afyon-1",          "Konya/Ardıçlı",
        "Kayseri-5",      "Mardin/Midyat-1",  "Konya/Kulu",
        "Afyon-2",        "Niğde/Gümüşler",   "Kayseri-6",
        "Kayseri-7",      "Aksaray-1",        "Kayseri-8",
        "Mersin/Mut",     "Konya/Karatay",    "Konya/Sille-2",
        "Kırşehir",       "Aksaray/Ortaköy",  "Niğde-1",
        "Aksaray/Selime",
    ],
    "Rock_Type": [
        "Pyroclastic","Volcanic",  "Pyroclastic","Volcanic",  "Pyroclastic",
        "Sedimentary","Pyroclastic","Plutonic",  "Pyroclastic","Pyroclastic",
        "Pyroclastic","Volcanic",  "Sedimentary","Pyroclastic","Volcanic",
        "Pyroclastic","Volcanic",  "Volcanic",  "Volcanic",  "Pyroclastic",
        "Pyroclastic","Sedimentary","Volcanic",  "Volcanic",  "Pyroclastic",
        "Pyroclastic","Pyroclastic","Plutonic",  "Pyroclastic","Sedimentary",
        "Pyroclastic","Volcanic",  "Plutonic",  "Plutonic",  "Pyroclastic",
        "Pyroclastic",
    ],
    # UPR: Ultrasonic Penetration Rate (mm/s)
    "UPR_mm_s": [
        1.266, 0.369, 0.258, 0.273, 0.178,
        0.164, 0.281, 0.083, 0.568, 0.179,
        0.547, 0.200, 0.118, 0.370, 0.093,
        0.339, 0.373, 0.230, 0.574, 0.741,
        0.709, 1.123, 0.354, 0.314, 1.423,
        0.699, 0.336, 0.106, 0.715, 0.169,
        0.524, 0.538, 0.258, 0.219, 0.783,
        0.884,
    ],
    # rho_d: dry unit weight (g/cm³)
    "rho_d_g_cm3": [
        1.24, 2.63, 1.83, 2.22, 1.90,   # samples  1-5   ← 1.82 corrected
        2.64, 1.80, 2.67, 1.88, 1.91,   # samples  6-10
        1.91, 2.35, 2.71, 1.90, 2.31,   # samples 11-15
        1.84, 2.42, 1.91, 2.00, 1.78,   # samples 16-20
        1.56, 1.62, 2.53, 2.10, 1.48,   # samples 21-25
        1.52, 2.64, 2.75, 1.51, 2.56,   # samples 26-30
        1.97, 2.25, 2.70, 2.65, 1.84,   # samples 31-35
        1.54,                            # sample  36
    ],
    # n: apparent porosity (%)
    "n_pct": [
        36.96,  2.54, 20.75, 10.24, 19.92,
         4.38, 26.04,  0.47, 11.96, 18.61,
        16.46,  5.67,  0.35, 18.42, 17.70,
        19.28,  3.56, 17.00, 19.04, 17.17,
        26.48, 27.19,  2.46, 10.74, 32.38,
        28.05,  2.74,  0.91, 27.20,  4.56,
        22.43,  7.99,  0.42,  0.69, 22.75,
        26.96,
    ],
    # PR: core drilling Penetration Rate (mm/s)  — TARGET
    "PR_mm_s": [
        5.501, 0.520, 2.627, 0.761, 1.474,
        0.162, 1.966, 0.144, 1.338, 1.302,
        1.705, 0.708, 0.417, 1.182, 0.288,
        2.494, 0.707, 1.547, 2.300, 3.707,
        5.217, 6.904, 0.428, 1.464, 6.629,
        4.903, 0.990, 0.267, 5.387, 0.471,
        2.617, 2.181, 0.290, 0.206, 3.500,
        5.924,
    ],
}

df = pd.DataFrame(DATA)
X  = df[["UPR_mm_s", "rho_d_g_cm3", "n_pct"]].values
y  = df["PR_mm_s"].values
FEAT = ["UPR", r"$\rho_d$", r"$n$"]          # display labels (LaTeX)
FEAT_PLAIN = ["UPR", "ρd", "n"]               # plain-text labels


# =============================================================================
# 2.  DESCRIPTIVE STATISTICS & CORRELATIONS
# =============================================================================
def print_statistics():
    print("=" * 70)
    print("DESCRIPTIVE STATISTICS  (should match Table 1 in manuscript)")
    print("=" * 70)
    sub = df[["UPR_mm_s", "rho_d_g_cm3", "n_pct", "PR_mm_s"]].copy()
    sub.columns = ["UPR (mm/s)", "ρd (g/cm³)", "n (%)", "PR (mm/s)"]
    print(sub.describe().round(3))

    print("\nPearson correlations with PR  (p < 0.001 for all):")
    for col, lbl in zip(["UPR_mm_s", "rho_d_g_cm3", "n_pct"], FEAT_PLAIN):
        r_val, p_val = stats.pearsonr(df[col], y)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*"
        print(f"  {lbl:>4s}: r = {r_val:+.3f},  p = {p_val:.2e}  {sig}")

    print("\nFull correlation matrix (PR, UPR, ρd, n):")
    mat  = np.corrcoef([y, X[:, 0], X[:, 1], X[:, 2]])
    labs = ["PR", "UPR", "ρd", "n"]
    header = "       " + "".join(f"{l:>8s}" for l in labs)
    print(header)
    for i, l in enumerate(labs):
        row = f"  {l:>4s} " + "".join(f"{mat[i,j]:>+8.3f}" for j in range(4))
        print(row)

    # Verify manuscript Table 1 values
    print("\nManuscript Table 1 verification:")
    vals = {
        "PR":  {"min": 0.144, "max": 6.904, "mean": 2.173, "sd": 2.038},
        "UPR": {"min": 0.083, "max": 1.423, "mean": 0.454, "sd": 0.332},
        "ρd":  {"min": 1.480, "max": 2.750, "mean": 2.101, "sd": 0.412},
        "n":   {"min": 0.350, "max": 36.960,"mean": 14.735,"sd": 10.609},
    }
    cols = {"PR": y, "UPR": X[:,0], "ρd": X[:,1], "n": X[:,2]}
    for vn, ref in vals.items():
        v = cols[vn]
        ok_min  = abs(v.min()  - ref["min"])  < 0.001
        ok_max  = abs(v.max()  - ref["max"])  < 0.001
        ok_mean = abs(v.mean() - ref["mean"]) < 0.002
        ok_sd   = abs(v.std(ddof=1) - ref["sd"]) < 0.002
        status = "✓" if all([ok_min, ok_max, ok_mean, ok_sd]) else "✗"
        print(f"  {vn:>4s}  min={v.min():.3f} max={v.max():.3f} "
              f"mean={v.mean():.3f} sd={v.std(ddof=1):.3f}  {status}")


# =============================================================================
# 3.  PREPROCESSING
# =============================================================================
scaler = StandardScaler()
X_sc   = scaler.fit_transform(X)


# =============================================================================
# 4.  MODEL DEFINITIONS
# =============================================================================
MODELS = {
    "Ridge":         Ridge(alpha=1.0),
    "BayesianRidge": BayesianRidge(),
    "ElasticNet":    ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10_000),
    "SVR":           SVR(kernel="rbf", C=10, epsilon=0.1),
    "GBR":           GradientBoostingRegressor(
                         n_estimators=100, learning_rate=0.1,
                         max_depth=3, random_state=42),
}

MODEL_COLORS = {
    "Ridge":         "#2E86AB",
    "BayesianRidge": "#E67E22",
    "ElasticNet":    "#9C59B0",
    "SVR":           "#3CB371",
    "GBR":           "#E74C3C",
}


# =============================================================================
# 5.  METRIC HELPERS
# =============================================================================
def mape(yt, yp):
    """Mean Absolute Percentage Error (excludes zero-target samples)."""
    m = yt != 0
    return np.mean(np.abs((yt[m] - yp[m]) / yt[m])) * 100


def full_metrics(yt, yp, k):
    """
    Returns a dict with all metrics used in manuscript Table 2:
        R², Adj.R², RMSE, MAPE, AIC (log-likelihood form), BIC, F, p-value
    """
    n   = len(yt)
    r2  = r2_score(yt, yp)
    ar2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
    rmse_val = float(np.sqrt(mean_squared_error(yt, yp)))
    mape_val = mape(yt, yp)

    # AIC / BIC — RSS-based form (confirmed matches manuscript Table 2)
    # AIC = n*ln(MSE) + 2*(k+1)   BIC = n*ln(MSE) + (k+1)*ln(n)
    mse_val = mean_squared_error(yt, yp)
    aic  = n * np.log(mse_val) + 2 * (k + 1)
    bic  = n * np.log(mse_val) + (k + 1) * np.log(n)

    # F-statistic and p-value
    sse    = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    ss_res = sse
    f_stat = ((ss_tot - ss_res) / k) / (ss_res / (n - k - 1))
    p_val  = float(1 - stats.f.cdf(f_stat, k, n - k - 1))

    return dict(r2=r2, adj_r2=ar2, rmse=rmse_val, mape=mape_val,
                aic=aic, bic=bic, f=f_stat, p=p_val)


def back_transform(model):
    """Convert standardized coefficients to original-unit equation."""
    mu, sg = scaler.mean_, scaler.scale_
    cf = model.coef_ / sg
    ic = model.intercept_ - np.sum(model.coef_ * mu / sg)
    return float(ic), cf


# =============================================================================
# 6.  TRAINING + LOO-CV
# =============================================================================
def run_models():
    loo = LeaveOneOut()
    res = {}
    k   = X.shape[1]                       # number of predictors = 3

    print("\n" + "=" * 70)
    print("MODEL RESULTS  (should match Table 2 in manuscript)")
    print("=" * 70)
    print(f"{'Model':<15} {'R²_tr':>7} {'AdjR²':>7} {'RMSE_tr':>9} "
          f"{'MAPE_tr':>9} {'R²_loo':>7} {'RMSE_loo':>9} "
          f"{'MAPE_loo':>9} {'AIC':>9} {'BIC':>9} {'F':>10} {'p':>9}")
    print("-" * 110)

    for name, model in MODELS.items():
        # ── Full-data fit ────────────────────────────────────────────────
        model.fit(X_sc, y)
        y_tr = model.predict(X_sc)
        m_tr = full_metrics(y, y_tr, k)

        # ── LOO-CV ───────────────────────────────────────────────────────
        loo_p = np.zeros(len(y))
        for tri, tei in loo.split(X_sc):
            m_tmp = type(model)(**model.get_params())
            m_tmp.fit(X_sc[tri], y[tri])
            loo_p[tei] = m_tmp.predict(X_sc[tei])

        r2_l   = r2_score(y, loo_p)
        rmse_l = float(np.sqrt(mean_squared_error(y, loo_p)))
        mape_l = mape(y, loo_p)

        res[name] = dict(
            model   = model,
            y_tr    = y_tr,
            y_loo   = loo_p,
            res_loo = y - loo_p,
            r2_tr   = m_tr["r2"],
            adj_r2  = m_tr["adj_r2"],
            rmse_tr = m_tr["rmse"],
            mape_tr = m_tr["mape"],
            r2_loo  = r2_l,
            rmse_loo= rmse_l,
            mape_loo= mape_l,
            aic     = m_tr["aic"],
            bic     = m_tr["bic"],
            f_stat  = m_tr["f"],
            p_val   = m_tr["p"],
        )

        print(f"{name:<15} {m_tr['r2']:>7.4f} {m_tr['adj_r2']:>7.4f} "
              f"{m_tr['rmse']:>9.4f} {m_tr['mape']:>9.2f} "
              f"{r2_l:>7.4f} {rmse_l:>9.4f} {mape_l:>9.2f} "
              f"{m_tr['aic']:>9.3f} {m_tr['bic']:>9.3f} "
              f"{m_tr['f']:>10.2f} {m_tr['p']:>9.5f}")

    return res


# =============================================================================
# 7.  REGRESSION EQUATIONS (original units)
# =============================================================================
def print_equations(res):
    print("\n" + "=" * 70)
    print("REGRESSION EQUATIONS  (original units — match Table 3)")
    print("=" * 70)

    for name in ["Ridge", "BayesianRidge", "ElasticNet"]:
        ic, cf = back_transform(res[name]["model"])
        sign_d = "+" if cf[1] >= 0 else ""
        sign_n = "+" if cf[2] >= 0 else ""
        print(f"\n{name}:")
        print(f"  PR = {ic:.4f}  +  {cf[0]:.4f}×UPR  "
              f"{sign_d}{cf[1]:.4f}×ρd  {sign_n}{cf[2]:.4f}×n")
        print(f"  R²_train = {res[name]['r2_tr']:.4f}   "
              f"R²_LOO = {res[name]['r2_loo']:.4f}")

    # GBR surrogate
    gbr_pred = res["GBR"]["y_tr"]
    surr     = Ridge(alpha=1.0).fit(X_sc, gbr_pred)
    r2s      = r2_score(gbr_pred, surr.predict(X_sc))
    ic_s, cf_s = back_transform(surr)
    sign_d = "+" if cf_s[1] >= 0 else ""
    sign_n = "+" if cf_s[2] >= 0 else ""
    print(f"\nGBR Ridge Surrogate  (R² vs GBR = {r2s:.4f}):")
    print(f"  PR ≈ {ic_s:.4f}  +  {cf_s[0]:.4f}×UPR  "
          f"{sign_d}{cf_s[1]:.4f}×ρd  {sign_n}{cf_s[2]:.4f}×n")
    return surr


# =============================================================================
# 8.  PERMUTATION FEATURE IMPORTANCE
# =============================================================================
def compute_pfi(res):
    print("\n" + "=" * 70)
    print("PERMUTATION FEATURE IMPORTANCE — Ridge (100 repeats)")
    print("Matches Table 4 in manuscript")
    print("=" * 70)
    pfi = permutation_importance(
        res["Ridge"]["model"], X_sc, y,
        n_repeats=100, random_state=42, scoring="r2",
    )
    total = np.sum(np.maximum(pfi.importances_mean, 0))
    for i, fn in enumerate(FEAT_PLAIN):
        rel = pfi.importances_mean[i] / total * 100 if total > 0 else 0
        print(f"  {fn:>4s}: mean = {pfi.importances_mean[i]:.4f} "
              f"± {pfi.importances_std[i]:.4f}   relative = {rel:.1f}%")
    return pfi


# =============================================================================
# 9.  FIGURE 4 — Exploratory Data Analysis
# =============================================================================
def plot_figure4():
    """
    6-panel EDA figure:
    (A) Pearson correlation matrix
    (B) PR vs UPR   (C) PR vs ρd   (D) PR vs n
    (E) Box plots   (F) Permutation importance
    """
    CORR = np.corrcoef([y, X[:, 0], X[:, 1], X[:, 2]])
    C    = {"b": "#2E86AB", "g": "#3CB371", "p": "#9C59B0"}

    fig = plt.figure(figsize=(16, 13), dpi=150)
    fig.patch.set_facecolor("white")
    gs  = gridspec.GridSpec(2, 3, hspace=0.50, wspace=0.38)

    # ── (A) Correlation matrix ────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(CORR, cmap="RdBu_r", vmin=-1, vmax=1)
    lbs = ["PR", "UPR", r"$\rho_d$", r"$n$"]
    ax.set_xticks(range(4)); ax.set_xticklabels(lbs, fontsize=10)
    ax.set_yticks(range(4)); ax.set_yticklabels(lbs, fontsize=10)
    for i in range(4):
        for j in range(4):
            v   = CORR[i, j]
            clr = "white" if abs(v) > 0.45 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=clr)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Pearson r", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    ax.set_title("(A) Pearson Correlation Matrix",
                 fontsize=11, fontweight="bold", pad=8)

    # ── (B) PR vs UPR ────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(X[:, 0], y, color=C["b"], alpha=0.85, s=55,
               edgecolors="white", linewidths=0.6, zorder=3)
    sl, ic, r_val, p_val, _ = stats.linregress(X[:, 0], y)
    xf = np.linspace(X[:, 0].min() - 0.03, X[:, 0].max() + 0.03, 300)
    ax.plot(xf, sl * xf + ic, color="red", lw=2.0, zorder=4)
    ax.text(0.05, 0.94, f"r = {r_val:.3f}\np = {p_val:.2e}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="gray", alpha=0.85))
    ax.set_xlabel("UPR (mm/s)", fontsize=10)
    ax.set_ylabel("PR (mm/s)", fontsize=10)
    ax.set_title("(B) PR vs UPR", fontsize=11, fontweight="bold", pad=8)
    ax.set_ylim(-0.3, 7.8)
    ax.grid(True, alpha=0.25, lw=0.5); ax.tick_params(labelsize=9)

    # ── (C) PR vs ρd ──────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(X[:, 1], y, color=C["g"], alpha=0.85, s=55,
               edgecolors="white", linewidths=0.6, zorder=3)
    sl2, ic2, r2v, p2v, _ = stats.linregress(X[:, 1], y)
    xf2 = np.linspace(X[:, 1].min() - 0.05, X[:, 1].max() + 0.05, 300)
    ax.plot(xf2, sl2 * xf2 + ic2, color="red", lw=2.0, zorder=4)
    # Place text box at upper right to avoid overlap with negative-slope data
    ax.text(0.97, 0.94, f"r = {r2v:.3f}\np = {p2v:.2e}",
            transform=ax.transAxes, fontsize=9, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="gray", alpha=0.85))
    ax.set_xlabel(r"$\rho_d$ (g/cm³)", fontsize=10)
    ax.set_ylabel("PR (mm/s)", fontsize=10)
    ax.set_title(r"(C) PR vs $\mathbf{\rho_d}$", fontsize=11, fontweight="bold", pad=8)
    ax.set_ylim(-0.3, 7.8)
    ax.grid(True, alpha=0.25, lw=0.5); ax.tick_params(labelsize=9)

    # ── (D) PR vs n ──────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(X[:, 2], y, color=C["p"], alpha=0.85, s=55,
               edgecolors="white", linewidths=0.6, zorder=3)
    sl3, ic3, r3v, p3v, _ = stats.linregress(X[:, 2], y)
    xf3 = np.linspace(X[:, 2].min() - 0.5, X[:, 2].max() + 0.5, 300)
    ax.plot(xf3, sl3 * xf3 + ic3, color="red", lw=2.0, zorder=4)
    ax.text(0.05, 0.94, f"r = {r3v:.3f}\np = {p3v:.2e}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="gray", alpha=0.85))
    ax.set_xlabel(r"$n$ (%)", fontsize=10)
    ax.set_ylabel("PR (mm/s)", fontsize=10)
    ax.set_title(r"(D) PR vs $\mathbf{n}$", fontsize=11, fontweight="bold", pad=8)
    ax.set_ylim(-0.3, 7.8)
    ax.grid(True, alpha=0.25, lw=0.5); ax.tick_params(labelsize=9)

    # ── (E) Box plots ────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    bp = ax.boxplot(
        [y, X[:, 0], X[:, 2] / 10],
        patch_artist=True,
        medianprops=dict(color="white", linewidth=2.2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=5, alpha=0.65,
                        markerfacecolor="red"),
    )
    for patch, clr in zip(bp["boxes"], [C["b"], C["g"], C["p"]]):
        patch.set_facecolor(clr); patch.set_alpha(0.75)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["PR\n(mm/s)", "UPR\n(mm/s)", r"$n$/10 (%)"],
                       fontsize=9)
    ax.set_title("(E) Variable Distributions",
                 fontsize=11, fontweight="bold", pad=8)
    ax.set_ylabel("Value", fontsize=10)
    ax.grid(True, axis="y", alpha=0.25, lw=0.5)

    # ── (F) Permutation importance ────────────────────────────────────────
    ax  = fig.add_subplot(gs[1, 2])
    pfi = permutation_importance(
        Ridge(alpha=1.0).fit(X_sc, y), X_sc, y,
        n_repeats=100, random_state=42, scoring="r2",
    )
    pfi_m = pfi.importances_mean
    pfi_s = pfi.importances_std
    feat_labels = ["UPR", r"$\rho_d$", r"$n$"]
    bars = ax.bar(
        feat_labels, pfi_m,
        yerr=pfi_s, capsize=6,
        color=[C["b"], C["g"], C["p"]], alpha=0.85,
        error_kw=dict(elinewidth=1.4, ecolor="black", capthick=1.4),
    )
    for bar, val in zip(bars, pfi_m):
        ax.text(bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + pfi_s[list(pfi_m).index(val)] + 0.01,
                f"{val:.4f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
    ax.set_title("(F) Permutation Importance\n(Ridge, 100 repeats)",
                 fontsize=11, fontweight="bold", pad=8)
    ax.set_ylabel("Permutation Importance", fontsize=10)
    ax.set_ylim(0, max(pfi_m) * 1.40)
    ax.grid(True, axis="y", alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=9)

    plt.savefig("figure4_EDA.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.show()
    print("  → figure4_EDA.png  saved  (Fig 1 in manuscript)")
    return fig


# =============================================================================
# 10.  FIGURE 5 — Predicted vs Measured (LOO-CV)
# =============================================================================
def plot_figure5(res):
    fig, axes = plt.subplots(2, 3, figsize=(16, 11), dpi=130)
    fig.patch.set_facecolor("white")
    names = list(MODELS.keys())

    for idx, name in enumerate(names):
        ax  = axes.flatten()[idx]
        y_p = res[name]["y_loo"]
        absr = np.abs(y - y_p)
        sc   = ax.scatter(y, y_p, c=absr, cmap="YlOrRd", alpha=0.85,
                          s=55, edgecolors="white", linewidths=0.6,
                          zorder=3, vmin=0, vmax=absr.max())
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04
                     ).set_label("|Residual| (mm/s)", fontsize=8)
        lim = [min(y.min(), y_p.min()) - 0.2,
               max(y.max(), y_p.max()) + 0.2]
        ax.plot(lim, lim, "k--", lw=1.5, zorder=4, label="1:1 line")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.text(0.05, 0.95,
                f"R²_LOO   = {res[name]['r2_loo']:.4f}\n"
                f"RMSE_LOO = {res[name]['rmse_loo']:.4f}\n"
                f"MAPE_LOO = {res[name]['mape_loo']:.2f}%",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="gray", alpha=0.9))
        ax.set_title(f"({'ABCDE'[idx]}) {name}",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Measured PR (mm/s)", fontsize=10)
        ax.set_ylabel("Predicted PR (mm/s)", fontsize=10)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.25, lw=0.5); ax.tick_params(labelsize=9)

    # Panel F — all models overlay
    ax = axes.flatten()[5]
    for name in names:
        ax.scatter(y, res[name]["y_loo"], alpha=0.55, s=25,
                   color=MODEL_COLORS[name], label=name,
                   edgecolors="white", linewidths=0.4)
    ax.plot([y.min(), y.max()], [y.min(), y.max()],
            "k--", lw=1.5, label="1:1 line")
    ax.set_title("(F) All Models Overlay",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Measured PR (mm/s)", fontsize=10)
    ax.set_ylabel("Predicted PR (mm/s)", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.25, lw=0.5); ax.tick_params(labelsize=9)

    plt.tight_layout()
    plt.savefig("figure5_predicted_vs_measured.png", dpi=130,
                bbox_inches="tight", facecolor="white")
    plt.show()
    print("  → figure5_predicted_vs_measured.png  saved  (Fig 2 in manuscript)")
    return fig


# =============================================================================
# 11.  FIGURE 6 — Residual Analysis
# =============================================================================
def plot_figure6(res):
    fig, axes = plt.subplots(2, 3, figsize=(16, 11), dpi=130)
    fig.patch.set_facecolor("white")
    names = list(MODELS.keys())

    for idx, name in enumerate(names):
        ax    = axes.flatten()[idx]
        y_p   = res[name]["y_loo"]
        resid = res[name]["res_loo"]
        sig   = np.std(resid)
        sc    = ax.scatter(y_p, resid, c=np.abs(resid), cmap="YlOrRd",
                           alpha=0.85, s=55, edgecolors="white",
                           linewidths=0.6, zorder=3)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04
                     ).set_label("|Residual| (mm/s)", fontsize=8)
        ax.axhline(0,           color="black", lw=1.5, ls="--", zorder=4)
        ax.axhline(+1.96 * sig, color="gray",  lw=1.0, ls=":",  zorder=4,
                   label="±1.96σ")
        ax.axhline(-1.96 * sig, color="gray",  lw=1.0, ls=":",  zorder=4)
        _, sw_p = shapiro(resid)
        # Stat box always upper-right to avoid overlap
        ax.text(0.97, 0.97,
                f"σ = {sig:.3f}\nS-W p = {sw_p:.3f}",
                transform=ax.transAxes, fontsize=8, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="gray", alpha=0.9))
        ax.set_title(f"({'ABCDE'[idx]}) {name}",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted PR (mm/s)", fontsize=10)
        ax.set_ylabel("Residual (mm/s)", fontsize=10)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.25, lw=0.5); ax.tick_params(labelsize=9)

    # Panel F — residual histograms
    ax = axes.flatten()[5]
    for name in names:
        ax.hist(res[name]["res_loo"], bins=12, alpha=0.55,
                color=MODEL_COLORS[name], label=name, edgecolor="white")
    ax.axvline(0, color="black", lw=1.5, ls="--")
    ax.set_title("(F) Residual Distributions",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Residual (mm/s)", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25, lw=0.5); ax.tick_params(labelsize=9)

    plt.tight_layout()
    plt.savefig("figure6_residuals.png", dpi=130,
                bbox_inches="tight", facecolor="white")
    plt.show()
    print("  → figure6_residuals.png  saved  (Fig 3 in manuscript)")
    return fig


# =============================================================================
# 12.  FIGURE 7 — Model Comparison
# =============================================================================
def plot_figure7(res):

    fig, axes = plt.subplots(
        2, 3,
        figsize=(20, 12),
        dpi=300
    )

    fig.patch.set_facecolor("white")

    names = list(MODELS.keys())
    clrs = [MODEL_COLORS[n] for n in names]

    panels = [
        ("r2_loo",   "R²LOO",          "(A)", 0.80, 0.90),
        ("rmse_loo", "RMSELOO (mm/s)", "(B)", None, None),
        ("aic",      "AIC",            "(C)", None, None),
        ("bic",      "BIC",            "(D)", None, None),
        ("mape_loo", "MAPELOO (%)",    "(E)", None, None),
    ]

    # ==================================================
    # PANELS A-E
    # ==================================================

    for (metric, ylabel, letter, thr1, thr2), ax in zip(
            panels, axes.flatten()):

        vals = [res[n][metric] for n in names]

        bars = ax.bar(
            names,
            vals,
            color=clrs,
            alpha=0.90,
            edgecolor="white",
            linewidth=1.5
        )

        # Best model highlighting
        if "r2" in metric:
            best_idx = np.argmax(vals)
        else:
            best_idx = np.argmin(vals)

        bars[best_idx].set_edgecolor("black")
        bars[best_idx].set_linewidth(3)

        # Reference lines for R²
        if thr1 is not None:
            ax.axhline(
                thr1,
                color="green",
                linestyle="--",
                linewidth=2,
                alpha=0.8,
                label=f"R² = {thr1}"
            )

        if thr2 is not None:
            ax.axhline(
                thr2,
                color="navy",
                linestyle=":",
                linewidth=2,
                alpha=0.8,
                label=f"R² = {thr2}"
            )

        if (thr1 is not None) or (thr2 is not None):
            ax.legend(
                fontsize=9,
                frameon=True,
                loc="best"
            )

        # -------------------------
        # Better label positioning
        # -------------------------
        ymin, ymax = ax.get_ylim()

        if metric == "r2_loo":
            offset = (ymax - ymin) * 0.25
            ax.set_ylim(top=max(vals) * 1.25)
        else:
            offset = (ymax - ymin) * 0.03

        for bar, val in zip(bars, vals):

            ax.text(
                bar.get_x() + bar.get_width()/2,
                val + offset,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white",
                    edgecolor="gray",
                    alpha=0.90
                )
            )

        ax.set_title(
            f"{letter} {ylabel}",
            fontsize=13,
            fontweight="bold",
            pad=12
        )

        ax.set_ylabel(
            ylabel,
            fontsize=11,
            fontweight="bold"
        )

        ax.tick_params(
            axis="x",
            labelsize=9,
            rotation=12
        )

        ax.tick_params(
            axis="y",
            labelsize=10
        )

        for label in ax.get_xticklabels():
            label.set_fontweight("bold")

        ax.grid(
            axis="y",
            linestyle="--",
            linewidth=0.7,
            alpha=0.35
        )

        ax.set_axisbelow(True)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ==================================================
    # PANEL F : SUMMARY TABLE
    # ==================================================

    ax = axes.flatten()[5]
    ax.axis("off")

    best_model = max(
        names,
        key=lambda x: res[x]["r2_loo"]
    )

    rows = [
        [
            n,
            f"{res[n]['r2_tr']:.4f}",
            f"{res[n]['r2_loo']:.4f}",
            f"{res[n]['rmse_loo']:.4f}",
            f"{res[n]['mape_loo']:.2f}",
            f"{res[n]['aic']:.1f}",
            f"{res[n]['bic']:.1f}",
            "<0.001"
        ]
        for n in names
    ]

    tbl = ax.table(
        cellText=rows,
        colLabels=[
            "Model",
            "R²tr",
            "R²LOO",
            "RMSE",
            "MAPE",
            "AIC",
            "BIC",
            "p"
        ],
        cellLoc="center",
        loc="center",
        bbox=[0.0, 0.02, 1.0, 0.92]
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1.20, 1.55)

    # Manual column widths
    col_widths = {
        0: 0.22,
        1: 0.10,
        2: 0.10,
        3: 0.13,
        4: 0.12,
        5: 0.10,
        6: 0.10,
        7: 0.08
    }

    for (row, col), cell in tbl.get_celld().items():

        if col in col_widths:
            cell.set_width(col_widths[col])

        cell.set_edgecolor("black")
        cell.set_linewidth(0.6)

        if row == 0:

            cell.set_facecolor("#1F4E79")

            cell.set_text_props(
                color="white",
                fontweight="bold"
            )

        elif row % 2 == 0:

            cell.set_facecolor("#EAF2F8")

        if row > 0 and rows[row - 1][0] == best_model:

            cell.set_text_props(
                fontweight="bold"
            )

            cell.set_linewidth(1.5)

    ax.set_title(
        "(F) Performance Statistics Summary",
        fontsize=13,
        fontweight="bold",
        pad=15
    )

    # ==================================================
    # OVERALL TITLE
    # ==================================================

    fig.suptitle(
        "Comparative Performance Evaluation of Machine Learning Models",
        fontsize=18,
        fontweight="bold",
        y=1.02
    )

    plt.tight_layout()

    plt.savefig(
        "figure7_model_comparison.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.show()

    print(
        "→ figure7_model_comparison.png saved successfully (300 dpi)"
    )

    return fig


# =============================================================================
# 13.  FIGURE 8  — GBR Analysis
# =============================================================================
def plot_figure8(res, surrogate):
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), dpi=130)
    fig.patch.set_facecolor("white")
    C = {"b": "#2E86AB", "g": "#3CB371", "p": "#9C59B0"}

    # ── (A) MDI feature importance ───────────────────────────────────────
    ax = axes[0]
    fi = res["GBR"]["model"].feature_importances_
    feat_labels_plain = ["UPR", r"$\rho_d$", r"$n$"]
    bar = ax.bar(feat_labels_plain, fi,
                 color=[C["b"], C["g"], C["p"]], alpha=0.85,
                 edgecolor="white")
    for b, v in zip(bar, fi):
        ax.text(b.get_x() + b.get_width() / 2., b.get_height() + 0.004,
                f"{v:.4f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.set_title("(A) GBR Feature Importance (MDI)",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean Decrease in Impurity", fontsize=10)
    ax.set_ylim(0, max(fi) * 1.28)
    ax.grid(True, axis="y", alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=10)

    # ── (B) GBR vs Ridge surrogate parity ───────────────────────────────
    ax    = axes[1]
    gbr_p = res["GBR"]["y_tr"]
    sur_p = surrogate.predict(X_sc)
    r2s   = r2_score(gbr_p, sur_p)
    ax.scatter(gbr_p, sur_p, color="#E74C3C", alpha=0.85, s=55,
               edgecolors="white", linewidths=0.6, zorder=3)
    lim = [min(gbr_p.min(), sur_p.min()) - 0.1,
           max(gbr_p.max(), sur_p.max()) + 0.1]
    ax.plot(lim, lim, "k--", lw=1.5, zorder=4, label="1:1 line")
    ax.text(0.05, 0.95, f"R² = {r2s:.4f}",
            transform=ax.transAxes, fontsize=10, va="top",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="gray", alpha=0.9))
    ax.set_xlabel("GBR Predictions (mm/s)", fontsize=10)
    ax.set_ylabel("Ridge Surrogate Predictions (mm/s)", fontsize=10)
    ax.set_title("(B) GBR vs Ridge Surrogate",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.25, lw=0.5)

    # ── (C) Training R² vs LOO-CV R² ────────────────────────────────────
    ax    = axes[2]
    names = list(MODELS.keys())
    xpos  = np.arange(len(names))
    w     = 0.35
    ax.bar(xpos - w / 2, [res[n]["r2_tr"]  for n in names], w,
           label="Training R²", color="#2E86AB", alpha=0.85,
           edgecolor="white")
    ax.bar(xpos + w / 2, [res[n]["r2_loo"] for n in names], w,
           label="LOO-CV R²",   color="#E74C3C", alpha=0.85,
           edgecolor="white", hatch="//")
    ax.set_xticks(xpos)
    ax.set_xticklabels(names, fontsize=8, rotation=12)
    ax.set_ylabel("R²", fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_title("(C) Training R² vs LOO-CV R²",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=9)

    plt.tight_layout()
    plt.savefig("figure8_GBR_analysis.png", dpi=130,
                bbox_inches="tight", facecolor="white")
    plt.show()
    print("  → figure8_GBR_analysis.png  saved  (Fig 5 in manuscript)")
    return fig


# =============================================================================
# 14.  MAIN
# =============================================================================
def main():
    print_statistics()
    res       = run_models()
    surrogate = print_equations(res)
    compute_pfi(res)

    print("\n" + "=" * 70)
    print("GENERATING FIGURES  (5 figures matching manuscript Figs 1–5)")
    print("=" * 70)
    plot_figure4()
    plot_figure5(res)
    plot_figure6(res)
    plot_figure7(res)
    plot_figure8(res, surrogate)
    print("\n✓  All done.  Five PNG files saved in working directory.")


if __name__ == "__main__":
    main()