import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, mannwhitneyu, spearmanr
import os

def validate_asset(gmsi, asset_df, asset_name):
    print(f"\n{'='*50}")
    print(f"Validating GMSI against {asset_name} Volatility")
    print(f"{'='*50}")
    
    # Merge on date
    df = pd.merge(gmsi[['date', 'pure_gmsi']], 
                  asset_df[['date', 'return', 'volatility_7d', 'volatility_14d', 'volatility_30d']], 
                  on='date', how='inner').dropna()
                  
    # Define Forward Targets
    # Forward 1d 'volatility' proxy is absolute return of the next day
    df['fwd_vol_1d'] = df['return'].abs().shift(-1)
    
    # Forward 7d volatility is the 7d rolling volatility 7 days from now
    df['fwd_vol_7d'] = df['volatility_7d'].shift(-7)
    
    # Forward 14d volatility is the 14d rolling volatility 14 days from now
    df['fwd_vol_14d'] = df['volatility_14d'].shift(-14)
    
    df = df.dropna()
    
    # 1. Lagged Correlations (Spearman to handle non-linearity and outliers)
    targets = {'1 Day': 'fwd_vol_1d', '7 Days': 'fwd_vol_7d', '14 Days': 'fwd_vol_14d'}
    
    print("\n--- 1. Lagged Correlations (GMSI(t) vs Forward Volatility) ---")
    real_corrs = {}
    for label, col in targets.items():
        corr, pval = spearmanr(df['pure_gmsi'], df[col])
        real_corrs[label] = corr
        print(f"GMSI vs Forward {label} Vol: {corr:.4f} (p-value: {pval:.4e})")
        
    # 2. Conditional Expectations (Quintiles)
    print("\n--- 2. Conditional Expectations ---")
    df['gmsi_quintile'] = pd.qcut(df['pure_gmsi'], 5, labels=['Q1(Low)', 'Q2', 'Q3', 'Q4', 'Q5(High)'])
    
    cond_exp = df.groupby('gmsi_quintile', observed=True)[['fwd_vol_7d', 'fwd_vol_14d']].mean()
    print("Mean Forward Volatility by GMSI Quintile:")
    print(cond_exp)
    
    # Plot Conditional Expectations
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cond_exp['fwd_vol_7d'].plot(kind='bar', ax=axes[0], color='skyblue')
    axes[0].set_title(f'{asset_name}: E[7d Vol | GMSI Quintile]')
    axes[0].set_ylabel('Mean Forward 7d Volatility')
    
    cond_exp['fwd_vol_14d'].plot(kind='bar', ax=axes[1], color='salmon')
    axes[1].set_title(f'{asset_name}: E[14d Vol | GMSI Quintile]')
    axes[1].set_ylabel('Mean Forward 14d Volatility')
    plt.tight_layout()
    plt.savefig(f'reports/figures/cond_exp_{asset_name}.png')
    
    # 3. Tail Event Analysis
    print("\n--- 3. Tail Event Analysis (Top 5% Volatility Days) ---")
    # Define top 5% most volatile 7d periods
    threshold_7d = df['fwd_vol_7d'].quantile(0.95)
    tail_events = df[df['fwd_vol_7d'] >= threshold_7d]
    normal_events = df[df['fwd_vol_7d'] < threshold_7d]
    
    print(f"Top 5% threshold for 7d vol: {threshold_7d:.4f} (N={len(tail_events)})")
    
    # Does GMSI predict these tail events? Compare GMSI(t) for tail vs normal
    ks_stat, ks_pval = ks_2samp(tail_events['pure_gmsi'], normal_events['pure_gmsi'])
    mw_stat, mw_pval = mannwhitneyu(tail_events['pure_gmsi'], normal_events['pure_gmsi'], alternative='two-sided')
    
    print(f"KS Test (Distributions differ) - Stat: {ks_stat:.4f}, p-value: {ks_pval:.4e}")
    print(f"Mann-Whitney U Test (Median shift) - Stat: {mw_stat:.4f}, p-value: {mw_pval:.4e}")
    
    # 4. Placebo Control (Empirical P-values)
    print("\n--- 4. Placebo Control (1000 Shuffles) ---")
    n_iterations = 1000
    np.random.seed(42)
    
    placebo_corrs = {label: [] for label in targets.keys()}
    
    for _ in range(n_iterations):
        shuffled_gmsi = np.random.permutation(df['pure_gmsi'].values)
        for label, col in targets.items():
            corr, _ = spearmanr(shuffled_gmsi, df[col])
            placebo_corrs[label].append(corr)
            
    # Compute Empirical P-values
    for label in targets.keys():
        real_c = real_corrs[label]
        placebo_dist = np.array(placebo_corrs[label])
        
        # Two-tailed empirical p-value
        empirical_pval = np.sum(np.abs(placebo_dist) >= np.abs(real_c)) / n_iterations
        print(f"{label} - Real Corr: {real_c:.4f} | Empirical p-value: {empirical_pval:.4f}")
        
    # Plot Placebo Distribution for 7 Days
    plt.figure(figsize=(8, 5))
    sns.histplot(placebo_corrs['7 Days'], bins=50, kde=True, color='gray', stat='density')
    plt.axvline(real_corrs['7 Days'], color='red', linestyle='dashed', linewidth=2, label=f"Real Corr: {real_corrs['7 Days']:.4f}")
    plt.title(f'Placebo Test Distribution vs True Correlation (7d Forward) - {asset_name}')
    plt.xlabel('Spearman Correlation')
    plt.legend()
    plt.savefig(f'reports/figures/placebo_test_{asset_name}.png')

def main():
    os.makedirs('reports/figures', exist_ok=True)
    
    # Load Pure GMSI
    gmsi = pd.read_csv('data/processed/gmsi_exogenous.csv')
    gmsi['date'] = pd.to_datetime(gmsi['date'])
    
    # Run for BTC
    btc = pd.read_csv('data/processed/btc_vsi_full.csv')
    btc['date'] = pd.to_datetime(btc['date'])
    validate_asset(gmsi, btc, 'BTC')
    
    # Run for NIFTY
    nifty = pd.read_csv('data/processed/nifty_vsi_full.csv')
    nifty['date'] = pd.to_datetime(nifty['date'])
    validate_asset(gmsi, nifty, 'NIFTY')

if __name__ == '__main__':
    main()
