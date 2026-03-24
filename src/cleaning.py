import pandas as pd
import numpy as np
import os

def run_qa_checks(df, stage="Raw"):
    """Requirement #2: Automated Data QA Reporting"""
    print(f"Running {stage} QA Checks...")
    
    report = []
    report.append(f"--- {stage} Data QA Report ---")
    
    # 1. Missingness
    missing = df.isnull().sum().sum()
    report.append(f"Missing Values: {missing} ({df.isnull().sum().max()} max in one column)")
    
    # 2. Duplicates
    dupes = df.index.duplicated().sum()
    report.append(f"Duplicate Timestamps: {dupes}")
    
    # 3. Outliers (Extreme Prices)
    outliers = df[(df['price'] > 500) | (df['price'] < -150)].shape[0]
    report.append(f"Extreme Price Outliers (>500 or <-150): {outliers}")
    
    # 4. Coverage (Time Gap Check)
    expected_hours = (df.index.max() - df.index.min()).total_seconds() / 3600 + 1
    actual_hours = len(df)
    coverage = (actual_hours / expected_hours) * 100
    report.append(f"Time Coverage: {coverage:.2f}% ({int(expected_hours - actual_hours)} hours missing)")

    # Save report to output folder
    os.makedirs('output', exist_ok=True)
    with open(f'output/qa_report_{stage.lower()}.txt', 'w') as f:
        f.write("\n".join(report))
    
    print(f"{stage} QA Complete. Report saved to output/qa_report_{stage.lower()}.txt")

def clean_and_engineer_features(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found!")
        return

    # Load
    df = pd.read_csv(input_path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    
    #  Raw QA 
    run_qa_checks(df, stage="Raw")

    # 1. Reindex & Interpolate (Fixing Coverage)
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='h', tz='UTC')
    df = df.reindex(full_range)
    df = df.interpolate(method='linear')
    
    # 2. Outlier Handling 
    df['Solar'] = df['Solar'].clip(lower=0)
    df['Wind_Onshore'] = df['Wind_Onshore'].clip(lower=0)
    df['Wind_Offshore'] = df['Wind_Offshore'].clip(lower=0)
    
    # 3. FEATURE ENGINEERING (Residual Load & Ratios)
    df_berlin = df.tz_convert('Europe/Berlin')
    
    # Core Fundamentals
    df_berlin['residual_load'] = df_berlin['load_actual'] - (df_berlin['Solar'] + df_berlin['Wind_Onshore'] + df_berlin['Wind_Offshore'])
    df_berlin['re_penetration'] = (df_berlin['Solar'] + df_berlin['Wind_Onshore'] + df_berlin['Wind_Offshore']) / df_berlin['load_actual']
    df_berlin['wind_share'] = (df_berlin['Wind_Onshore'] + df_berlin['Wind_Offshore']) / df_berlin['load_actual']
    df_berlin['solar_share'] = df_berlin['Solar'] / df_berlin['load_actual']
    df_berlin['hour_cos'] = np.cos(2 * np.pi * df_berlin.index.hour / 24) # Needed for peak impact
    df_berlin['solar_peak_impact'] = df_berlin['Solar'] * df_berlin['hour_cos']

    # Lags & Time features
    df_berlin['price_lag_24h'] = df_berlin['price'].shift(24)
    df_berlin['price_lag_168h'] = df_berlin['price'].shift(168)
    df_berlin['hour_sin'] = np.sin(2 * np.pi * df_berlin.index.hour / 24)
    df_berlin['is_weekend'] = df_berlin.index.dayofweek.isin([5, 6]).astype(int)

    # Processed QA 
    df_final = df_berlin.dropna()
    run_qa_checks(df_final, stage="Processed")

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path)
    print(f"Cleaning & QA Complete. Processed data saved to {output_path}")

if __name__ == "__main__":
    clean_and_engineer_features('data/raw_data.csv', 'data/processed_data.csv')