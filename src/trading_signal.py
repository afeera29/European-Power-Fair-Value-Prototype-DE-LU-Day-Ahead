import pandas as pd
import os
import json

def generate_trading_signal(forecast_path, market_price_assumption=95.0):
    if not os.path.exists(forecast_path):
        print(f"Error: {forecast_path} not found. Run model.py first!")
        return

    print(f"Analyzing Signal (Market Assumption: €{market_price_assumption})...")
    df = pd.read_csv(forecast_path, index_col=0, parse_dates=True)
    
    # Baseload calculation
    fair_value_baseload = float(df['pred_xgboost'].mean())
    edge = fair_value_baseload - market_price_assumption
    
    print(f"\n--- TRADING SIGNAL REPORT ---")
    print(f"Model Fair Value (Baseload): €{fair_value_baseload:.2f}/MWh")
    print(f"Market Forward Price:         €{market_price_assumption:.2f}/MWh")
    print(f"Calculated Edge:              €{edge:.2f}/MWh")
    
    if edge < -5.0:
        signal = "SELL (SHORT) THE WEEK-AHEAD"
        logic = "Market is overpricing the week; model expects fundamental weakness."
    elif edge > 5.0:
        signal = "BUY (LONG) THE WEEK-AHEAD"
        logic = "Market is underpricing scarcity; model expects fundamental tightness."
    else:
        signal = "NEUTRAL"
        logic = "Price is within fair value range."
    
    # --- Save results to JSON for AI Agent ---
    summary_data = {
        "fair_value": fair_value_baseload,
        "market_price": market_price_assumption,
        "edge": edge,
        "signal": signal
    }
    
    os.makedirs('output', exist_ok=True)
    with open('output/trading_summary.json', 'w') as f:
        json.dump(summary_data, f)
    print("Handover data saved to output/trading_summary.json")
    # ----------------------------------------------

    print(f"SIGNAL: {signal}")
    print(f"LOGIC:  {logic}\n")

if __name__ == "__main__":
    generate_trading_signal('output/final_forecasts.csv')