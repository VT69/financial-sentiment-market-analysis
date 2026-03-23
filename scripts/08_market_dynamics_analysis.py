"""
=============================================================================
NOTEBOOK 08: Market Dynamics Analysis
=============================================================================
Objective:
    Understand WHY and HOW markets move through:
    1. Market Fragility Index (MFI)
    2. Shock Propagation Analysis
    3. Regime-Conditioned Behavior

Philosophy:
    - No price-derived inputs in GMSI (already corrected)
    - No look-ahead / future leakage
    - No black-box ML
    - Weak but real signals > strong but artificial ones
    - Every metric has a clear mathematical definition

Data Expected (from your project's data/processed/ folder):
    - btc_features.csv  : BTC daily data with log_returns, vol_7d, vol_14d, vol_30d
    - nifty_features.csv: NIFTY daily data with same structure
    - gmsi_daily.csv    : Pure exogenous GMSI (no price-derived inputs)

If files are not found, realistic synthetic data is generated for demo purposes.
=============================================================================
"""

# ─── 0. Imports ──────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import warnings
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal
import os

warnings.filterwarnings("ignore")
np.random.seed(42)

plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#c9d1d9",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#c9d1d9",
    "grid.color":       "#21262d",
    "grid.linewidth":   0.6,
    "figure.titlesize": 14,
    "axes.titlesize":   12,
    "font.family":      "monospace",
})

COLORS = {
    "low":    "#3fb950",   # green  – low stress / healthy
    "medium": "#d29922",   # yellow – medium
    "high":   "#f85149",   # red    – high stress / fragile
    "btc":    "#f7931a",   # bitcoin orange
    "nifty":  "#0070f3",   # blue   – nifty
    "accent": "#bc8cff",   # purple – highlights
    "mfi":    "#79c0ff",   # light blue – MFI
}

SAVE_DIR = "market_dynamics_output"
os.makedirs(SAVE_DIR, exist_ok=True)


# ─── 1. Data Loading / Synthetic Fallback ────────────────────────────────────

def load_or_generate_data():
    """
    Attempt to load real project data.
    Falls back to realistic synthetic data if files are missing.

    Synthetic data is calibrated to match:
        BTC:   annualised vol ~80%, fat-tailed returns, volatility clustering
        NIFTY: annualised vol ~20%, moderate clustering
        GMSI:  exogenous, bounded [0,1], slightly AR(1)
    """
    DATA_PATHS = {
        "btc":   ["data/processed/btc_features.csv",
                  "data/raw/btc_daily.csv",
                  "notebooks/data/btc_features.csv"],
        "nifty": ["data/processed/nifty_features.csv",
                  "data/raw/nifty_daily.csv",
                  "notebooks/data/nifty_features.csv"],
        "gmsi":  ["data/processed/gmsi_daily.csv",
                  "src/gmsi_daily.csv",
                  "notebooks/data/gmsi_daily.csv"],
    }

    results = {}
    for key, paths in DATA_PATHS.items():
        loaded = False
        for p in paths:
            if os.path.exists(p):
                try:
                    df = pd.read_csv(p, parse_dates=True, index_col=0)
                    df.index = pd.to_datetime(df.index)
                    results[key] = df
                    print(f"  ✅ Loaded {key} from: {p}  ({len(df)} rows)")
                    loaded = True
                    break
                except Exception as e:
                    print(f"  ⚠️  Could not parse {p}: {e}")
        if not loaded:
            print(f"  🔧 Generating synthetic {key} data (demo mode)")
            results[key] = None

    return results


def simulate_garch_returns(n, omega, alpha, beta, mu=0.0, df_t=5):
    """
    Simulate returns from GARCH(1,1) with t-distributed innovations.
    This gives realistic volatility clustering and fat tails.

    σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
    r_t  = μ + σ_t · z_t,   z_t ~ t(df)

    Parameters are in daily log-return units:
        BTC:   ω~1e-4 gives ~σ_daily ≈ 3-5%
        NIFTY: ω~2e-5 gives ~σ_daily ≈ 1%
    """
    sigma2    = np.zeros(n)
    ret       = np.zeros(n)
    uncond_v  = omega / (1 - alpha - beta + 1e-10)
    sigma2[0] = uncond_v   # unconditional variance (start at steady state)

    for t in range(1, n):
        eps2       = ret[t-1] ** 2
        sigma2[t]  = omega + alpha * eps2 + beta * sigma2[t-1]
        sigma2[t]  = max(sigma2[t], 1e-10)
        # t-distribution scaled so variance = 1: divide by sqrt(df/(df-2))
        scale      = np.sqrt(df_t / (df_t - 2))
        z          = np.random.standard_t(df_t) / scale
        ret[t]     = mu + np.sqrt(sigma2[t]) * z

    return ret, np.sqrt(sigma2)


def build_features(ret, sigma, dates):
    """
    Build log returns + rolling volatility features from simulated data.
    All returns and vol are in natural log-return space (e.g., 0.03 = 3%).
    """
    df = pd.DataFrame({"log_return": ret, "realized_vol_daily": sigma}, index=dates)

    # Annualised rolling historical volatility (standard in finance)
    df["vol_7d"]  = df["log_return"].rolling(7).std()  * np.sqrt(252)
    df["vol_14d"] = df["log_return"].rolling(14).std() * np.sqrt(252)
    df["vol_30d"] = df["log_return"].rolling(30).std() * np.sqrt(252)

    # Absolute return in daily log-return units (used for shock detection)
    df["abs_return"] = df["log_return"].abs()

    return df.dropna()


def generate_synthetic_data():
    """
    Generate 2000 days (~8 years) of realistic synthetic market + GMSI data.
    """
    n     = 2000
    dates = pd.bdate_range("2016-01-01", periods=n)

    # BTC: high vol (~3-5% daily), strong clustering (α+β close to 1), fat tails
    # ω = 1e-5, uncond. daily vol = sqrt(1e-5 / (1-0.12-0.85)) = sqrt(0.000333) ≈ 1.8%
    btc_ret, btc_sig = simulate_garch_returns(
        n, omega=1e-5, alpha=0.12, beta=0.85, mu=0.0003, df_t=4
    )
    btc = build_features(btc_ret, btc_sig, dates)

    # NIFTY: lower vol (~1% daily), moderate clustering
    # ω = 3e-6, uncond. daily vol ≈ sqrt(0.0001) = 1%
    nifty_ret, nifty_sig = simulate_garch_returns(
        n, omega=3e-6, alpha=0.07, beta=0.90, mu=0.0003, df_t=6
    )
    nifty = build_features(nifty_ret, nifty_sig, dates)

    # GMSI: purely exogenous AR(1) with beta regime-shifts
    # Slight negative relationship with future vol (as found in research)
    gmsi_raw = np.zeros(n)
    gmsi_raw[0] = 0.4
    for t in range(1, n):
        shock = np.random.normal(0, 0.08)
        gmsi_raw[t] = 0.3 + 0.65 * gmsi_raw[t-1] + shock
        # Regime jumps (simulate news spikes)
        if np.random.rand() < 0.02:
            gmsi_raw[t] += np.random.choice([-0.3, 0.3])

    # Clip to [0,1] and smooth slightly
    gmsi_vals = pd.Series(gmsi_raw).clip(0, 1).rolling(3, min_periods=1).mean().values
    gmsi = pd.DataFrame({"gmsi": gmsi_vals}, index=dates)

    return btc, nifty, gmsi


# ─── 2. Market Fragility Index (MFI) ─────────────────────────────────────────

def compute_mfi(df, window=30):
    """
    Market Fragility Index (MFI)
    ─────────────────────────────
    A composite, interpretable metric of how sensitive the market is to shocks.

    Components (all computed over a rolling window, no future leakage):

    (A) Volatility Persistence  = AC1(|r_t|)
        Measured as the lag-1 autocorrelation of absolute returns over [window] days.
        High AC1 → shocks do not dissipate quickly → fragile.

    (B) Vol-of-Vol (VoV)
        = std(rolling_vol_7d) / mean(rolling_vol_7d)  over [window] days
        High VoV → volatility regime is unstable → fragile.

    (C) Tail Risk Frequency
        = fraction of days in [window] where |r_t| > 2σ (rolling 30d σ)
        Captures how often large moves occur, independent of their sign.

    MFI = (A_norm + B_norm + C_norm) / 3
    Each component is min-max normalised over its own full history
    (using expanding min/max to avoid look-ahead).

    Interpretation:
        MFI → 1 : Market is fragile; shocks tend to amplify
        MFI → 0 : Market is resilient; shocks decay quickly
    """
    results = pd.DataFrame(index=df.index)

    # ── Component A: Volatility Persistence (AC1 of |returns|) ──────────────
    abs_ret = df["log_return"].abs()

    def rolling_ac1(series, w):
        """Lag-1 autocorrelation of a rolling window."""
        vals = []
        for i in range(len(series)):
            if i < w - 1:
                vals.append(np.nan)
            else:
                chunk = series.iloc[i - w + 1: i + 1].values
                if np.std(chunk) < 1e-10:
                    vals.append(0.0)
                else:
                    corr = np.corrcoef(chunk[:-1], chunk[1:])[0, 1]
                    vals.append(np.nan_to_num(corr, nan=0.0))
        return pd.Series(vals, index=series.index)

    results["persistence"] = rolling_ac1(abs_ret, window)

    # ── Component B: Vol-of-Vol ──────────────────────────────────────────────
    vol7 = df["vol_7d"]
    results["vol_of_vol"] = (
        vol7.rolling(window).std() /
        vol7.rolling(window).mean().replace(0, np.nan)
    )

    # ── Component C: Tail Risk Frequency ────────────────────────────────────
    # 2σ threshold from 30-day rolling std (pure historical, no leakage)
    sigma_30 = df["log_return"].rolling(30).std()
    is_tail  = (abs_ret > 2 * sigma_30).astype(float)
    results["tail_freq"] = is_tail.rolling(window).mean()

    # ── Normalise each component (expanding min-max, no look-ahead) ─────────
    for col in ["persistence", "vol_of_vol", "tail_freq"]:
        s = results[col]
        expanding_min = s.expanding().min()
        expanding_max = s.expanding().max()
        denom = (expanding_max - expanding_min).replace(0, np.nan)
        results[f"{col}_norm"] = (s - expanding_min) / denom

    # ── Composite MFI ────────────────────────────────────────────────────────
    norm_cols = ["persistence_norm", "vol_of_vol_norm", "tail_freq_norm"]
    results["MFI"] = results[norm_cols].mean(axis=1)

    return results


# ─── 3. Shock Propagation Analysis ───────────────────────────────────────────

def identify_shocks(df, quantile=0.95):
    """
    Define shocks as days where |log_return| exceeds the [quantile] threshold.
    Uses the expanding empirical quantile to avoid look-ahead bias.

    Returns boolean Series: True on shock days.
    """
    abs_ret = df["log_return"].abs()
    # Expanding quantile: threshold on day t uses only data up to t-1
    threshold = abs_ret.shift(1).expanding().quantile(quantile)
    shocks    = abs_ret > threshold
    return shocks, threshold


def compute_shock_propagation(df, shock_mask, horizons=(1, 3, 7, 14, 21)):
    """
    For each shock day, measure forward realized volatility at t+h.

    Forward vol at horizon h = abs(log_return at t+h)
    (Single-day absolute return as a proxy for instantaneous vol;
     rolling average is also computed for smoothness.)

    Returns:
        shock_df: DataFrame of shock events with forward vol at each horizon
        summary : mean ± std forward vol by horizon
    """
    abs_ret  = df["log_return"].abs()
    shock_idx = np.where(shock_mask.values)[0]

    records = []
    for idx in shock_idx:
        row = {
            "shock_date":    df.index[idx],
            "shock_mag":     abs_ret.iloc[idx],
            "shock_date_idx": idx,
        }
        for h in horizons:
            fwd_idx = idx + h
            if fwd_idx < len(df):
                row[f"fwd_vol_t{h}"] = abs_ret.iloc[fwd_idx]
                # 3-day centred rolling vol around t+h (if available)
                lo = max(0, fwd_idx - 1)
                hi = min(len(df), fwd_idx + 2)
                row[f"fwd_rollingvol_t{h}"] = abs_ret.iloc[lo:hi].mean()
            else:
                row[f"fwd_vol_t{h}"]        = np.nan
                row[f"fwd_rollingvol_t{h}"] = np.nan
        records.append(row)

    shock_df = pd.DataFrame(records).set_index("shock_date")

    # Summary: mean forward vol by horizon
    fwd_cols  = [f"fwd_vol_t{h}" for h in horizons]
    means     = shock_df[fwd_cols].mean()
    stds      = shock_df[fwd_cols].std()
    summary   = pd.DataFrame({"horizon": list(horizons),
                               "mean_fwd_vol": means.values,
                               "std_fwd_vol":  stds.values})
    return shock_df, summary


def estimate_shock_halflife(summary):
    """
    Approximate shock half-life by fitting an exponential decay to
    mean forward vol vs horizon.

    Model: V(h) = V_0 · exp(-λ·h)
    Half-life = ln(2) / λ

    Returns: (lambda, half_life_days, R²)
    """
    h_vals = np.array(summary["horizon"], dtype=float)
    v_vals = np.array(summary["mean_fwd_vol"], dtype=float)

    # Remove NaN
    mask   = ~np.isnan(v_vals)
    h_vals = h_vals[mask]
    v_vals = v_vals[mask]

    if len(h_vals) < 3:
        return np.nan, np.nan, np.nan

    # Log-linear regression: log(V) = log(V_0) - λ·h
    log_v = np.log(v_vals + 1e-10)
    slope, intercept, r, p, se = stats.linregress(h_vals, log_v)
    lam       = -slope
    half_life = np.log(2) / lam if lam > 0 else np.inf
    r2        = r ** 2

    return lam, half_life, r2


# ─── 4. Regime Conditioning ──────────────────────────────────────────────────

def define_regimes(gmsi_series, low_q=0.20, high_q=0.80):
    """
    Define market stress regimes from GMSI.

    Regime assignment uses expanding quantiles (no look-ahead):
        Low  Stress: GMSI ≤ 20th percentile
        High Stress: GMSI ≥ 80th percentile
        Medium     : in between

    Returns: Series of regime labels {'low', 'medium', 'high'}
    """
    regimes = pd.Series("medium", index=gmsi_series.index, dtype=str)

    for i in range(1, len(gmsi_series)):
        hist = gmsi_series.iloc[:i]
        lo   = hist.quantile(low_q)
        hi   = hist.quantile(high_q)
        val  = gmsi_series.iloc[i]
        if val <= lo:
            regimes.iloc[i] = "low"
        elif val >= hi:
            regimes.iloc[i] = "high"
        # else stays "medium"

    return regimes


def regime_stats(df, regimes, mfi_df):
    """
    Compute key statistics within each regime:
        - Mean / std of realized volatility
        - Mean MFI
        - Shock frequency
        - Shock persistence (mean AC1 of abs returns)
    """
    combined = df[["log_return", "vol_30d", "abs_return"]].copy()
    combined["regime"] = regimes
    combined["MFI"]    = mfi_df["MFI"]

    stats_records = []
    for regime in ["low", "medium", "high"]:
        mask = combined["regime"] == regime
        sub  = combined[mask]
        if len(sub) < 10:
            continue

        # Volatility persistence within regime (AC1 of abs returns)
        ac1 = sub["abs_return"].autocorr(lag=1)

        stats_records.append({
            "regime":        regime,
            "n_days":        len(sub),
            "mean_vol30d":   sub["vol_30d"].mean(),
            "std_vol30d":    sub["vol_30d"].std(),
            "mean_MFI":      sub["MFI"].mean(),
            "shock_freq":    (sub["abs_return"] > sub["abs_return"].quantile(0.95)).mean(),
            "persistence_ac1": ac1,
        })

    return pd.DataFrame(stats_records).set_index("regime")


# ─── 5. Plotting Functions ────────────────────────────────────────────────────

def fig_mfi_over_time(df, mfi_df, regimes, asset_name):
    """Figure 1: MFI time series with regime background shading."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                             gridspec_kw={"hspace": 0.08})
    fig.suptitle(f"Market Fragility Index — {asset_name}", y=0.98, fontsize=14, color="#c9d1d9")

    dates = df.index

    # ── Panel 1: Price proxy (cumulative log return) ──
    ax1 = axes[0]
    cum_ret = df["log_return"].cumsum().values
    ax1.plot(dates, cum_ret, color=COLORS["btc"] if "BTC" in asset_name else COLORS["nifty"],
             linewidth=0.9, alpha=0.9)
    ax1.set_ylabel("Cum. Log Return", fontsize=9)
    ax1.set_title("Price Index (Cumulative Log Return)", fontsize=10, pad=4)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: MFI ──
    ax2 = axes[1]
    mfi = mfi_df["MFI"].reindex(dates)
    ax2.plot(dates, mfi, color=COLORS["mfi"], linewidth=1.0, alpha=0.85, label="MFI")
    ax2.fill_between(dates, mfi, alpha=0.15, color=COLORS["mfi"])
    ax2.axhline(0.7, color=COLORS["high"], linestyle="--", linewidth=0.8, alpha=0.6, label="High fragility (0.7)")
    ax2.axhline(0.3, color=COLORS["low"],  linestyle="--", linewidth=0.8, alpha=0.6, label="Low fragility (0.3)")
    ax2.set_ylabel("MFI", fontsize=9)
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Vol 30d with regime shading ──
    ax3 = axes[2]
    vol30 = df["vol_30d"].reindex(dates)
    ax3.plot(dates, vol30, color="#c9d1d9", linewidth=0.8, alpha=0.85)

    # Regime background shading
    reg_colors = {"low": COLORS["low"], "medium": COLORS["medium"], "high": COLORS["high"]}
    reg_vals   = regimes.reindex(dates).fillna("medium")
    prev_regime = None
    seg_start   = None
    for d, r in reg_vals.items():
        if r != prev_regime:
            if prev_regime is not None:
                ax3.axvspan(seg_start, d, alpha=0.12, color=reg_colors[prev_regime], linewidth=0)
            seg_start   = d
            prev_regime = r
    if prev_regime:
        ax3.axvspan(seg_start, dates[-1], alpha=0.12, color=reg_colors[prev_regime], linewidth=0)

    ax3.set_ylabel("30d Volatility", fontsize=9)
    ax3.set_xlabel("Date", fontsize=9)
    patches = [Patch(color=COLORS["low"],    alpha=0.5, label="Low Stress"),
               Patch(color=COLORS["medium"], alpha=0.5, label="Medium Stress"),
               Patch(color=COLORS["high"],   alpha=0.5, label="High Stress")]
    ax3.legend(handles=patches, fontsize=8, loc="upper right")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    path = f"{SAVE_DIR}/fig1_mfi_{asset_name.lower()}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    return fig


def fig_shock_decay(shock_summary_list, labels):
    """Figure 2: Shock propagation / decay curves per asset."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = [COLORS["btc"], COLORS["nifty"], COLORS["accent"]]

    for i, (summary, label) in enumerate(zip(shock_summary_list, labels)):
        h    = summary["horizon"].values
        mean = summary["mean_fwd_vol"].values
        std  = summary["std_fwd_vol"].values
        c    = colors[i % len(colors)]
        ax.plot(h, mean, "o-", color=c, label=label, linewidth=1.5, markersize=5)
        ax.fill_between(h, mean - 0.5*std, mean + 0.5*std, alpha=0.15, color=c)

    ax.set_xlabel("Days After Shock", fontsize=10)
    ax.set_ylabel("Mean Forward Absolute Return", fontsize=10)
    ax.set_title("Shock Propagation: Forward Volatility Decay After Top-5% Events", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    path = f"{SAVE_DIR}/fig2_shock_decay.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    return fig


def fig_regime_shock_response(df, shock_df, regimes, asset_name, horizons=(1, 3, 7, 14)):
    """
    Figure 3: Shock response separated by regime.
    Bar chart: mean forward vol at each horizon, grouped by regime.
    """
    # Align regimes to shock events
    shock_dates   = shock_df.index
    shock_regimes = regimes.reindex(shock_dates).fillna("medium")

    fig, axes = plt.subplots(1, len(horizons), figsize=(14, 5), sharey=True)
    fig.suptitle(f"Shock Response by Regime — {asset_name}", fontsize=13, color="#c9d1d9")

    regime_order  = ["low", "medium", "high"]
    regime_colors = [COLORS["low"], COLORS["medium"], COLORS["high"]]

    for i, h in enumerate(horizons):
        ax  = axes[i]
        col = f"fwd_vol_t{h}"
        if col not in shock_df.columns:
            continue

        means = []
        stds  = []
        ns    = []
        for r in regime_order:
            mask   = shock_regimes == r
            vals   = shock_df.loc[mask, col].dropna()
            means.append(vals.mean() if len(vals) > 0 else 0)
            stds.append(vals.std()   if len(vals) > 1 else 0)
            ns.append(len(vals))

        bars = ax.bar(regime_order, means, color=regime_colors, alpha=0.75,
                      edgecolor="#30363d", linewidth=0.5)
        ax.errorbar(regime_order, means, yerr=[s/2 for s in stds],
                    fmt="none", color="#c9d1d9", capsize=4, linewidth=1.2)

        # Add N labels
        for bar, n in zip(bars, ns):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                    f"n={n}", ha="center", va="bottom", fontsize=7.5, color="#8b949e")

        ax.set_title(f"t+{h} days", fontsize=10)
        ax.set_xlabel("Regime", fontsize=8)
        if i == 0:
            ax.set_ylabel("Mean |Return|", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = f"{SAVE_DIR}/fig3_regime_shock_{asset_name.lower()}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    return fig


def fig_mfi_by_regime(regime_stats_df, asset_name):
    """Figure 4: MFI distribution across regimes (violin/box-style summary)."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    fig.suptitle(f"Regime Statistics — {asset_name}", fontsize=13, color="#c9d1d9")

    metrics = [
        ("mean_vol30d",   "Mean Volatility (30d)"),
        ("mean_MFI",      "Mean MFI"),
        ("persistence_ac1", "Vol Persistence (AC1)"),
    ]

    regime_order  = [r for r in ["low", "medium", "high"] if r in regime_stats_df.index]
    regime_colors = {"low": COLORS["low"], "medium": COLORS["medium"], "high": COLORS["high"]}

    for ax, (col, title) in zip(axes, metrics):
        vals   = [regime_stats_df.loc[r, col] if r in regime_stats_df.index else 0
                  for r in regime_order]
        colors = [regime_colors[r] for r in regime_order]
        bars   = ax.bar(regime_order, vals, color=colors, alpha=0.8,
                        edgecolor="#30363d", linewidth=0.5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Regime", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() * 1.02 if v >= 0 else bar.get_height() - abs(v)*0.05,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8.5, color="#c9d1d9")

    plt.tight_layout()
    path = f"{SAVE_DIR}/fig4_regime_stats_{asset_name.lower()}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    return fig


def fig_mfi_components(mfi_df, asset_name):
    """Figure 5: Individual MFI component time series (decomposition)."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True,
                             gridspec_kw={"hspace": 0.08})
    fig.suptitle(f"MFI Component Decomposition — {asset_name}", fontsize=13, color="#c9d1d9")

    components = [
        ("persistence_norm", "A: Volatility Persistence (AC1 of |r|)", COLORS["btc"]),
        ("vol_of_vol_norm",  "B: Vol-of-Vol (CoV of 7d vol)",          COLORS["accent"]),
        ("tail_freq_norm",   "C: Tail Risk Frequency (|r| > 2σ)",      COLORS["high"]),
    ]

    for ax, (col, label, color) in zip(axes, components):
        if col in mfi_df.columns:
            ax.plot(mfi_df.index, mfi_df[col], color=color, linewidth=0.9, alpha=0.85)
            ax.fill_between(mfi_df.index, mfi_df[col], alpha=0.12, color=color)
        ax.set_ylabel(label, fontsize=8.5)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Date", fontsize=9)
    plt.tight_layout()
    path = f"{SAVE_DIR}/fig5_mfi_components_{asset_name.lower()}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    return fig


# ─── 6. Statistical Tests ─────────────────────────────────────────────────────

def regime_significance_tests(df, regimes, col="vol_30d"):
    """
    Test whether distributions of [col] differ significantly across regimes.
    Uses Kruskal-Wallis (non-parametric, appropriate for financial data).
    Also reports Mann-Whitney U for pairwise comparisons.
    """
    print(f"\n  Statistical Significance Tests for '{col}' across regimes:")
    print("  " + "─" * 55)

    groups = {}
    for r in ["low", "medium", "high"]:
        mask     = regimes == r
        groups[r] = df.loc[mask, col].dropna().values

    # Kruskal-Wallis
    try:
        stat, p = kruskal(*[groups[r] for r in groups if len(groups[r]) > 0])
        print(f"  Kruskal-Wallis H={stat:.3f}, p={p:.4f}  "
              f"{'✅ Significant' if p < 0.05 else '❌ Not significant'}")
    except Exception as e:
        print(f"  Kruskal-Wallis failed: {e}")

    # Pairwise Mann-Whitney
    pairs = [("low", "high"), ("low", "medium"), ("medium", "high")]
    for r1, r2 in pairs:
        if len(groups.get(r1, [])) < 3 or len(groups.get(r2, [])) < 3:
            continue
        try:
            stat, p = mannwhitneyu(groups[r1], groups[r2], alternative="two-sided")
            print(f"  Mann-Whitney [{r1} vs {r2}]: U={stat:.0f}, p={p:.4f}  "
                  f"{'✅' if p < 0.05 else '❌'}")
        except Exception as e:
            print(f"  Mann-Whitney [{r1} vs {r2}] failed: {e}")


# ─── 7. Main Analysis Pipeline ────────────────────────────────────────────────

def run_analysis(df_market, gmsi_df, asset_name):
    """
    Full analysis pipeline for one asset.

    Steps:
      1. Align market data with GMSI
      2. Compute MFI
      3. Define regimes
      4. Identify shocks
      5. Shock propagation
      6. Regime conditioning
      7. Statistical tests
      8. Plots
    Returns: dict of all computed artifacts
    """
    print(f"\n{'='*65}")
    print(f"  ANALYSIS: {asset_name}")
    print(f"{'='*65}")
    print(f"  Market data:  {len(df_market)} rows  ({df_market.index[0].date()} → {df_market.index[-1].date()})")

    # ── Step 1: Align ────────────────────────────────────────────────────────
    common_idx = df_market.index.intersection(gmsi_df.index)
    if len(common_idx) < 100:
        print(f"  ⚠️  Only {len(common_idx)} overlapping dates with GMSI — using market-only dates")
        # Forward-fill GMSI or use neutral value if no overlap
        gmsi_aligned = pd.Series(0.5, index=df_market.index, name="gmsi")
    else:
        gmsi_aligned = gmsi_df["gmsi"].reindex(df_market.index).ffill()
        print(f"  Aligned:      {len(common_idx)} dates")

    # ── Step 2: MFI ──────────────────────────────────────────────────────────
    print(f"\n  [1/5] Computing Market Fragility Index...")
    mfi_df = compute_mfi(df_market, window=30)
    print(f"        MFI range: [{mfi_df['MFI'].min():.3f}, {mfi_df['MFI'].max():.3f}]")
    print(f"        Mean MFI : {mfi_df['MFI'].mean():.3f}")

    # ── Step 3: Regimes ──────────────────────────────────────────────────────
    print(f"\n  [2/5] Defining Regimes from GMSI...")
    regimes = define_regimes(gmsi_aligned)
    rc      = regimes.value_counts()
    for r in ["low", "medium", "high"]:
        n = rc.get(r, 0)
        print(f"        {r:8s}: {n:5d} days ({100*n/len(regimes):.1f}%)")

    # ── Step 4: Shocks ────────────────────────────────────────────────────────
    print(f"\n  [3/5] Identifying Shocks (top 5% abs returns)...")
    shock_mask, threshold = identify_shocks(df_market, quantile=0.95)
    n_shocks = shock_mask.sum()
    print(f"        Shocks identified: {n_shocks}  ({100*n_shocks/len(df_market):.1f}% of days)")

    # ── Step 5: Propagation ───────────────────────────────────────────────────
    print(f"\n  [4/5] Computing Shock Propagation...")
    horizons       = (1, 3, 7, 14, 21)
    shock_df, summary = compute_shock_propagation(df_market, shock_mask, horizons=horizons)
    lam, halflife, r2  = estimate_shock_halflife(summary)
    print(f"        Shock half-life  : {halflife:.1f} days  (λ={lam:.4f}, R²={r2:.3f})")
    print(f"        Forward vol at t+1:  {summary.loc[summary.horizon==1,'mean_fwd_vol'].values[0]:.4f}")
    print(f"        Forward vol at t+14: {summary.loc[summary.horizon==14,'mean_fwd_vol'].values[0]:.4f}")

    # ── Step 6: Regime Stats ──────────────────────────────────────────────────
    print(f"\n  [5/5] Regime-Conditioned Statistics...")
    rstats = regime_stats(df_market, regimes, mfi_df)
    print(rstats.to_string())

    # ── Statistical Tests ──────────────────────────────────────────────────────
    df_aligned = df_market.copy()
    df_aligned["regime"] = regimes
    regime_significance_tests(df_aligned, regimes, col="vol_30d")
    regime_significance_tests(df_aligned, regimes, col="abs_return")

    # ── Plots ──────────────────────────────────────────────────────────────────
    print(f"\n  Generating plots...")
    plt.close("all")
    fig1 = fig_mfi_over_time(df_market, mfi_df, regimes, asset_name)
    fig2 = fig_shock_decay([summary], [asset_name])    # single asset; multi-asset done later
    fig3 = fig_regime_shock_response(df_market, shock_df, regimes, asset_name)
    fig4 = fig_mfi_by_regime(rstats, asset_name)
    fig5 = fig_mfi_components(mfi_df, asset_name)
    plt.close("all")

    return {
        "asset":    asset_name,
        "df":       df_market,
        "mfi_df":   mfi_df,
        "regimes":  regimes,
        "shock_df": shock_df,
        "summary":  summary,
        "rstats":   rstats,
        "halflife": halflife,
        "lambda":   lam,
    }


def run_comparative_shock_plot(results_list):
    """Figure 6: Side-by-side shock decay for all assets."""
    summaries = [r["summary"] for r in results_list]
    labels    = [r["asset"]   for r in results_list]
    fig       = fig_shock_decay(summaries, labels)
    path      = f"{SAVE_DIR}/fig6_shock_decay_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close("all")


def print_interpretation(results):
    """Print plain-English scientific interpretation of results."""
    r = results
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  SCIENTIFIC INTERPRETATION — {r['asset']:<33} ║
╚══════════════════════════════════════════════════════════════════╝

1. MARKET FRAGILITY INDEX (MFI)
   ─────────────────────────────
   The MFI captures three independent dimensions of fragility:
     A. Volatility Persistence: high AC1 of |returns| → shocks self-reinforce
     B. Vol-of-Vol: erratic volatility → regime instability
     C. Tail Frequency: clustering of large moves → non-Gaussian risk

   Mean MFI = {r['mfi_df']['MFI'].mean():.3f}
   → {'Elevated fragility. Market was frequently in states where '
      'shocks could amplify.' if r['mfi_df']['MFI'].mean() > 0.55
      else 'Moderate fragility. Most of the sample was in recoverable regimes.'}

2. SHOCK PROPAGATION
   ──────────────────
   Shock half-life ≈ {r['halflife']:.1f} days
   → Shocks {'persist for several weeks — evidence of slow information '
              'absorption or liquidity constraints.' if r['halflife'] > 10
              else 'decay within roughly 1–2 weeks — markets absorb news relatively fast.'}

   This is NOT prediction — it describes the structural dynamics of how
   information and risk flow through the market.

3. REGIME BEHAVIOR
   ─────────────────
   Key cross-regime differences:
{r['rstats'][['mean_vol30d','mean_MFI','persistence_ac1']].to_string()}

   Interpretation:
   → If MFI is HIGHER in low-stress regimes: markets become complacent
     and accumulate hidden fragility during quiet periods. Shocks then
     hit hard precisely because participants are under-positioned for risk.
   → If MFI is HIGHER in high-stress regimes: markets remain aware of
     their fragility and may actually recover faster (vol compression
     after risk-off events).

4. CORE INSIGHT
   ─────────────
   Markets are not driven directly by stress or news content, but by
   how information INTERACTS with current fragility levels, positioning,
   and liquidity. A given shock will have very different impact depending
   on the structural state the market is in when it arrives.
""")


# ─── 8. Entry Point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*65)
    print("  MARKET DYNAMICS ANALYSIS")
    print("  Phase 2: Fragility · Shock Propagation · Regime Dynamics")
    print("="*65)

    # ── Load data ──────────────────────────────────────────────────────────
    print("\n[LOADING DATA]")
    raw = load_or_generate_data()

    # ── Resolve: use real data if available, else synthetic ────────────────
    if all(v is None for v in raw.values()):
        print("\n  All files missing → generating full synthetic dataset")
        btc_df, nifty_df, gmsi_df = generate_synthetic_data()
    else:
        btc_df, nifty_df, gmsi_df = raw.get("btc"), raw.get("nifty"), raw.get("gmsi")

        if gmsi_df is None:
            print("  ⚠️  GMSI not found — generating synthetic GMSI")
            n     = max(len(btc_df) if btc_df is not None else 0,
                        len(nifty_df) if nifty_df is not None else 0)
            dates = (btc_df if btc_df is not None else nifty_df).index
            _, _, gmsi_df = generate_synthetic_data()
            gmsi_df = gmsi_df.reindex(dates).ffill().bfill()

        if btc_df is None:
            print("  ⚠️  BTC data not found — generating synthetic BTC")
            btc_df, _, _ = generate_synthetic_data()

        if nifty_df is None:
            print("  ⚠️  NIFTY data not found — generating synthetic NIFTY")
            _, nifty_df, _ = generate_synthetic_data()

        # Ensure required columns exist
        for df, name in [(btc_df, "BTC"), (nifty_df, "NIFTY")]:
            if "log_return" not in df.columns and "Close" in df.columns:
                df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
                print(f"  Computed log_return for {name} from Close prices")
            if "abs_return" not in df.columns:
                df["abs_return"] = df["log_return"].abs()
            for w in [7, 14, 30]:
                col = f"vol_{w}d"
                if col not in df.columns:
                    df[col] = df["log_return"].rolling(w).std() * np.sqrt(252)
            df.dropna(subset=["log_return"], inplace=True)

    # ── Run analysis ───────────────────────────────────────────────────────
    print("\n[RUNNING ANALYSIS]")
    btc_results   = run_analysis(btc_df.copy(),   gmsi_df, asset_name="BTC")
    nifty_results = run_analysis(nifty_df.copy(), gmsi_df, asset_name="NIFTY")

    # ── Comparative plot ───────────────────────────────────────────────────
    print("\n[COMPARATIVE PLOTS]")
    run_comparative_shock_plot([btc_results, nifty_results])

    # ── Interpretations ────────────────────────────────────────────────────
    print_interpretation(btc_results)
    print_interpretation(nifty_results)

    print(f"\n{'='*65}")
    print(f"  OUTPUT FILES → {SAVE_DIR}/")
    print(f"  fig1_mfi_*.png          : MFI time series + regime background")
    print(f"  fig2_shock_decay.png    : Forward vol decay after shocks")
    print(f"  fig3_regime_shock_*.png : Shock response by regime")
    print(f"  fig4_regime_stats_*.png : Regime-level summary statistics")
    print(f"  fig5_mfi_components_*   : MFI component decomposition")
    print(f"  fig6_shock_decay_*.png  : BTC vs NIFTY shock decay comparison")
    print(f"{'='*65}\n")
