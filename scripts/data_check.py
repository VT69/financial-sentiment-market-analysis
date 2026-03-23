import pandas as pd

def check_file(path, name):
    print(f'\n--- {name} ---')
    try:
        df = pd.read_csv(path)
        print(f'Columns: {df.columns.tolist()}')
        print(f'Row count: {len(df)}')
        
        date_col = 'date' if 'date' in df.columns else ('Date' if 'Date' in df.columns else None)
        if date_col:
            print(f'Date range: {df[date_col].min()} to {df[date_col].max()}')
    except Exception as e:
        print(f'Error: {e}')

check_file('data/processed/btc_vsi_long_v2.csv', 'BTC Data')
check_file('data/processed/nifty_vsi_long_v2.csv', 'NIFTY Data')
check_file('data/processed/text_with_sentiment.csv', 'Sentiment Data (Text)')
check_file('data/processed/btc_sentiment_aligned.csv', 'BTC Sentiment Data')
check_file('data/processed/daily_events.csv', 'Daily Events Data')
