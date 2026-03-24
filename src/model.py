import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import os
import numpy as np
import shap

def run_modeling_comparison(data_path):
    print("Starting Expanding Window Walk-Forward Validation ...")
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run cleaning.py first!")
        return

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # 1. Features and Target 
    features = [
        'residual_load', 're_penetration', 'price_lag_24h', 
        'price_lag_168h', 'hour_sin', 'hour_cos', 'is_weekend',
        'wind_share', 'solar_share', 'solar_peak_impact' # New Additive Features
    ]
    target = 'price'

    # 2. TimeSeriesSplit 
    tscv = TimeSeriesSplit(n_splits=5)
    
    fold_maes = []
    
    print("\n--- CROSS-VALIDATION RESULTS ---")
    for i, (train_index, test_index) in enumerate(tscv.split(df)):
        train, test = df.iloc[train_index], df.iloc[test_index]
        
        # Train Model
        model = xgb.XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=6)
        model.fit(train[features], train[target])
        
        # Predict
        preds = model.predict(test[features])
        mae = mean_absolute_error(test[target], preds)
        fold_maes.append(mae)
        
        print(f"Fold {i+1}: MAE = {mae:.2f} EUR/MWh (Train size: {len(train)} hrs)")

    # 3. Final Run
    split_idx = int(len(df) * 0.8)
    train_final, test_final = df.iloc[:split_idx], df.iloc[split_idx:].copy()
    
    final_model = xgb.XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=6)
    final_model.fit(train_final[features], train_final[target])
    
    test_final['pred_baseline'] = test_final['price_lag_168h']
    test_final['pred_xgboost'] = final_model.predict(test_final[features])
    
    # 4. Performance & Tail Risk Metrics 
    avg_mae = np.mean(fold_maes)
    final_mae = mean_absolute_error(test_final[target], test_final['pred_xgboost'])
    mae_baseline = mean_absolute_error(test_final[target], test_final['pred_baseline'])
    
    # Tail Risk Calculation 
    absolute_errors = np.abs(test_final[target] - test_final['pred_xgboost'])
    tail_risk_95 = np.percentile(absolute_errors, 95)

    print("\n--- FINAL PERFORMANCE METRICS ---")
    print(f"Avg CV MAE (Expanding Window): {avg_mae:.2f} EUR/MWh")
    print(f"Final XGBoost MAE:           {final_mae:.2f} EUR/MWh")
    print(f"Tail Risk (95th Pct Error): {tail_risk_95:.2f} EUR/MWh")
    print(f"Performance Uplift:          {((mae_baseline - avg_mae) / mae_baseline) * 100:.1f}%\n")

    # 4. TAIL RISK METRICS 
    # Calculating the absolute difference between Actual and Predicted
    test_errors = np.abs(test_final[target] - test_final['pred_xgboost'])
    
    # 95th Percentile Error:
    tail_risk_95 = np.percentile(test_errors, 95)
    
    # Max Error
    max_error = np.max(test_errors)

    print("\n--- RISK MANAGEMENT METRICS ---")
    print(f"Tail Risk (95th Pct Error): {tail_risk_95:.2f} EUR/MWh")
    print(f"Maximum Observed Error:    {max_error:.2f} EUR/MWh")
    
    # Logging this to a text file for the "Risk Desk"
    with open('output/risk_audit.txt', 'w') as f:
        f.write(f"Tail Risk (P95): {tail_risk_95:.2f} EUR/MWh\n")
        f.write(f"Max Error: {max_error:.2f} EUR/MWh\n")

    # 5. Saving Results for Trading Signal
    os.makedirs('output', exist_ok=True)
    test_final.to_csv('output/final_forecasts.csv')
    
    # 6. Create submission.csv
    submission = test_final[['pred_xgboost']].reset_index()
    submission.columns = ['id', 'y_pred']
    submission.to_csv('output/submission.csv', index=False)

    # 7. Save Plot
    plt.figure(figsize=(12, 6))
    plt.plot(test_final.index[:168], test_final[target][:168], label='Actual', color='black')
    plt.plot(test_final.index[:168], test_final['pred_xgboost'][:168], label='XGBoost', color='green')
    plt.legend()
    plt.title('Final Forecast Performance (Walk-Forward Validation)')
    plt.savefig('output/model_comparison.png')
    
    print("Pipeline complete. Results saved for Trading Desk.")

    # 8. SHAP Explainability 
    print("Generating SHAP explanations for the Trading Desk...")
    
    # Create the SHAP Explainer
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(test_final[features])

    # Save a Summary Plot 
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, test_final[features], show=False)
    plt.title("Feature Importance (SHAP) - Market Drivers")
    plt.savefig('output/shap_summary.png')
    plt.close()

    # Save the mean SHAP values to a CSV
    shap_df = pd.DataFrame({
        'feature': features,
        'importance': np.abs(shap_values).mean(0)
    }).sort_values(by='importance', ascending=False)
    
    shap_df.to_csv('output/feature_importance_audit.csv', index=False)
    
    print("SHAP analysis complete. Summary saved to output/shap_summary.png")

if __name__ == "__main__":
    run_modeling_comparison('data/processed_data.csv')