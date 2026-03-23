import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, mannwhitneyu, skew, kurtosis
from statsmodels.tsa.stattools import acf
import os

def load_data():
    gmsi = pd.read_csv('data/processed/gmsi_exogenous.csv')
    gmsi['date'] = pd.to_datetime(gmsi['date'])
    
    btc = pd.read_csv('data/processed/btc_vsi_full.csv')
    btc['date'] = pd.to_datetime(btc['date'])
    
    nifty = pd.read_csv('data/processed/nifty_vsi_full.csv')
    nifty['date'] = pd.to_datetime(nifty['date'])
    
    return gmsi, btc, nifty

def assign_regimes(gmsi):
    """
    Step 1: Regime Construction
    - Low Stress: bottom 20%
    - Medium Stress: 20–80%
    - High Stress: top 20%
    """
    p20 = gmsi['pure_gmsi'].quantile(0.20)
    p80 = gmsi['pure_gmsi'].quantile(0.80)
    
    conditions = [
        (gmsi['pure_gmsi'] <= p20),
        (gmsi['pure_gmsi'] > p20) & (gmsi['pure_gmsi'] <= p80),
        (gmsi['pure_gmsi'] > p80)
    ]
    choices = ['Low Stress', 'Medium Stress', 'High Stress']
    
    gmsi['Regime'] = np.select(conditions, choices, default='Medium Stress')
    # Ensure ordered categorical for plotting
    gmsi['Regime'] = pd.Categorical(gmsi['Regime'], categories=['Low Stress', 'Medium Stress', 'High Stress'], ordered=True)
    return gmsi

def regime_diagnostics(df, asset_name):
    """
    Step 2: Regime Diagnostics
    """
    print(f"\n{'*'*50}")
    print(f"REGIME DIAGNOSTICS: {asset_name}")
    print(f"{'*'*50}")
    
    regimes = ['Low Stress', 'Medium Stress', 'High Stress']
    results = []
    
    for r in regimes:
        regime_data = df[df['Regime'] == r].copy()
        
        # Mean & Median
        mean_7d = regime_data['volatility_7d'].mean()
        med_7d = regime_data['volatility_7d'].median()
        mean_14d = regime_data['volatility_14d'].mean()
        med_14d = regime_data['volatility_14d'].median()
        
        # Vol-of-vol (Standard deviation of volatility)
        vov_7d = regime_data['volatility_7d'].std()
        
        # Distribution shape
        skew_7d = skew(regime_data['volatility_7d'].dropna())
        kurt_7d = kurtosis(regime_data['volatility_7d'].dropna())
        
        # Persistence AR(1)
        # To compute AR(1) accurately, we need continuous sequences. 
        # But for an approximation of persistence *within* a regime, we can compute ACF(1) on the volatility series 
        # dropping NaNs within that regime slice (though it breaks time continuity, it shows state stickiness).
        # A more rigorous way is to calculate ACF on the whole series and just note it, 
        # but to see if Vol is MORE persistent in high stress, we'll approximate AR(1).
        if len(regime_data.dropna(subset=['volatility_7d'])) > 10:
            ar1 = acf(regime_data['volatility_7d'].dropna(), nlags=1)[1]
        else:
            ar1 = np.nan
            
        results.append({
            'Regime': r,
            'N_days': len(regime_data),
            'Mean_7d': mean_7d,
            'Median_7d': med_7d,
            'Mean_14d': mean_14d,
            'Median_14d': med_14d,
            'Vol_of_Vol_7d': vov_7d,
            'Skewness_7d': skew_7d,
            'Kurtosis_7d': kurt_7d,
            'AR(1)_7d': ar1
        })
        
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    return res_df

def shock_response(df, asset_name):
    """
    Step 3: Shock Response (Local Projections)
    """
    print(f"\n--- Shock Response Analysis: {asset_name} ---")
    
    # Define a volatility shock as top 5% absolute return days
    df['abs_return'] = df['return'].abs()
    shock_threshold = df['abs_return'].quantile(0.95)
    
    # Identify shock days
    shock_days = df[df['abs_return'] >= shock_threshold].copy()
    print(f"Number of shock days (Top 5% abs return): {len(shock_days)}")
    
    # We want to condition on the GMSI Regime of the day *prior* (t-1) to the shock to avoid leakage.
    # Actually, GMSI(t) is exogenous, but let's just use GMSI(t-1) regime to be safe.
    df['Regime_t_minus_1'] = df['Regime'].shift(1)
    
    # Calculate forward volatility for shocks
    results = []
    regimes = ['Low Stress', 'Medium Stress', 'High Stress']
    
    for r in regimes:
        # Shocks that occurred when prior day was in regime r
        regime_shocks = df[(df['abs_return'] >= shock_threshold) & (df['Regime_t_minus_1'] == r)]
        
        # We want the average of Volatility at t+1, t+7, t+14 relative to t.
        # Volatility is usually computed backwards (vol_7d is t-6 to t).
        # So forward vol at t+k is vol_7d shifted by -k
        fwd_1d_mean = pd.Series((regime_shocks.index + 1).map(lambda x: df.loc[x, 'volatility_7d'] if x in df.index else np.nan)).mean()
        fwd_7d_mean = pd.Series((regime_shocks.index + 7).map(lambda x: df.loc[x, 'volatility_7d'] if x in df.index else np.nan)).mean()
        fwd_14d_mean = pd.Series((regime_shocks.index + 14).map(lambda x: df.loc[x, 'volatility_7d'] if x in df.index else np.nan)).mean()

        results.append({
            'Regime': r,
            'Shock_Count': len(regime_shocks),
            'Fwd_Vol_t+1': fwd_1d_mean,
            'Fwd_Vol_t+7': fwd_7d_mean,
            'Fwd_Vol_t+14': fwd_14d_mean
        })
        
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    return res_df

def tail_risk_conditioning(df, asset_name):
    """
    Step 4: Tail Risk Conditioning
    - Compute P(Volatility in top 5% | GMSI regime).
    - Perform KS tests.
    """
    print(f"\n--- Tail Risk Conditioning: {asset_name} ---")
    
    # Define top 5% volatility tail event (looking at forward 7d volatility to test prediction)
    # Alternatively, look at concurrent 7d volatility conditional on regime.
    # User said: "Compute P(Volatility in top 5% | GMSI regime)."
    # Assume contemporaneous or forward state. Let's use contemporaneous for pure regime association.
    
    vol_tail_threshold = df['volatility_7d'].quantile(0.95)
    df['is_tail'] = (df['volatility_7d'] >= vol_tail_threshold).astype(int)
    
    overall_prob = df['is_tail'].mean()
    print(f"Unconditional P(Tail Volatility) = {overall_prob:.4f}")
    
    regimes = ['Low Stress', 'Medium Stress', 'High Stress']
    
    for r in regimes:
        regime_data = df[df['Regime'] == r]
        prob = regime_data['is_tail'].mean()
        print(f"P(Tail Volatility | {r}) = {prob:.4f}  (N={len(regime_data)})")
        
    # Statistical test: Does Volatility distribution differ between High and Low stress?
    low_vol = df[df['Regime'] == 'Low Stress']['volatility_7d'].dropna()
    high_vol = df[df['Regime'] == 'High Stress']['volatility_7d'].dropna()
    
    ks_stat, ks_pval = ks_2samp(high_vol, low_vol)
    mw_stat, mw_pval = mannwhitneyu(high_vol, low_vol, alternative='two-sided')
    
    print("\nStatistical Differences in Volatility (High vs Low Stress):")
    print(f"KS Test Statistic: {ks_stat:.4f} (p-value: {ks_pval:.4e})")
    print(f"Mann-Whitney U Test: {mw_stat:.4f} (p-value: {mw_pval:.4e})")

def generate_visualizations(df_btc, df_nifty):
    print("\nGenerating Visualizations...")
    os.makedirs('reports/figures/regime', exist_ok=True)
    
    # 1. Boxplots of volatility by regime
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.boxplot(x='Regime', y='volatility_7d', data=df_btc, ax=axes[0], palette='Blues')
    axes[0].set_title('BTC: 7d Volatility by GMSI Regime')
    sns.boxplot(x='Regime', y='volatility_7d', data=df_nifty, ax=axes[1], palette='Reds')
    axes[1].set_title('NIFTY: 7d Volatility by GMSI Regime')
    plt.tight_layout()
    plt.savefig('reports/figures/regime/boxplot_vol_by_regime.png')
    
    # 2. ACF Overlay (We calculate ACF for up to 20 lags for the whole series, but segmented)
    # This is tricky because slicing breaks time. A safer plot is Heatmap.
    
    # 3. Heatmap: GMSI regime x Volatility Percentile
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    def plot_heatmap(df, ax, title, cmap):
        # Create quantiles for Volatility
        df['Vol_Quintile'] = pd.qcut(df['volatility_7d'], 5, labels=['Q1(Low)', 'Q2', 'Q3', 'Q4', 'Q5(High)'])
        crosstab = pd.crosstab(df['Regime'], df['Vol_Quintile'], normalize='index') * 100
        sns.heatmap(crosstab, annot=True, fmt=".1f", cmap=cmap, ax=ax, cbar_kws={'label': '% of Regime'})
        ax.set_title(title)
        
    plot_heatmap(df_btc, axes[0], 'BTC: GMSI Regime vs Volatility Quintile', 'Blues')
    plot_heatmap(df_nifty, axes[1], 'NIFTY: GMSI Regime vs Volatility Quintile', 'Reds')
    plt.tight_layout()
    plt.savefig('reports/figures/regime/heatmap_regime_vs_vol.png')
    
    print("Saved plots to reports/figures/regime/")

def main():
    gmsi, btc, nifty = load_data()
    gmsi = assign_regimes(gmsi)
    
    # Merge regime labels into asset data
    df_btc = pd.merge(btc, gmsi[['date', 'Regime', 'pure_gmsi']], on='date', how='inner').dropna(subset=['volatility_7d'])
    df_nifty = pd.merge(nifty, gmsi[['date', 'Regime', 'pure_gmsi']], on='date', how='inner').dropna(subset=['volatility_7d'])
    
    # BTC Analysis
    regime_diagnostics(df_btc, "BTC")
    shock_df_btc = shock_response(df_btc, "BTC")
    tail_risk_conditioning(df_btc, "BTC")
    
    # NIFTY Analysis
    regime_diagnostics(df_nifty, "NIFTY")
    shock_df_nifty = shock_response(df_nifty, "NIFTY")
    tail_risk_conditioning(df_nifty, "NIFTY")
    
    # Shock Response Curves Validation (Plotting)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    def plot_shock(ax, shock_df, title):
        df_plot = shock_df.set_index('Regime')[['Fwd_Vol_t+1', 'Fwd_Vol_t+7', 'Fwd_Vol_t+14']].T
        df_plot.plot(marker='o', ax=ax)
        ax.set_title(title)
        ax.set_ylabel('Mean Forward 7d Volatility')
        ax.set_xlabel('Time Horizon post-Shock')
        ax.set_xticks(range(3))
        ax.set_xticklabels(['t+1', 't+7', 't+14'])
        
    plot_shock(axes[0], shock_df_btc, 'BTC: Shock Response by Regime')
    plot_shock(axes[1], shock_df_nifty, 'NIFTY: Shock Response by Regime')
    plt.tight_layout()
    os.makedirs('reports/figures/regime', exist_ok=True)
    plt.savefig('reports/figures/regime/shock_response.png')
    
    # Visualizations
    generate_visualizations(df_btc, df_nifty)
    
if __name__ == '__main__':
    main()
