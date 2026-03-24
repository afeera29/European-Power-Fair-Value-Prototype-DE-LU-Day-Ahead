from entsoe import EntsoePandasClient
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("ENTSOE_TOKEN") 

def fetch_entsoe_data():
    client = EntsoePandasClient(api_key=API_KEY)
    
    # pulling data 
    start = pd.Timestamp('20250101', tz='UTC') 
    end = pd.Timestamp.now(tz='UTC')
    country_code = 'DE_LU'

    print(f"Connecting to ENTSO-E for {country_code}...")

    try:
        # Step 1: Fetch all data
        print("Step 1/3: Fetching Day-Ahead Prices...")
        prices = client.query_day_ahead_prices(country_code, start=start, end=end)
        
        print("Step 2/3: Fetching Actual Load...")
        load = client.query_load(country_code, start=start, end=end)
        
        print("Step 3/3: Fetching Generation (Wind/Solar)...")
        gen = client.query_generation(country_code, start=start, end=end)

        # Assembly
        print("Assembling master dataframe...")
        
        # Initialize with Price Index
        df = pd.DataFrame(index=prices.index)
        df['price'] = prices
        
        # Handle Load 
        df['load_actual'] = load.iloc[:, 0] if isinstance(load, pd.DataFrame) else load

        # Handle Generation Technologies using a Search and Extract method
        # looking for the technology name in the columns and take the first match
        for target_col, search_key in [
            ('Solar', 'Solar'), 
            ('Wind_Onshore', 'Wind Onshore'), 
            ('Wind_Offshore', 'Wind Offshore')
        ]:
            if search_key in gen.columns.get_level_values(0):
                # taking the first sub-column (usually 'Actual Aggregated') for that tech
                df[target_col] = gen[search_key].iloc[:, 0].reindex(df.index, method='ffill')
            else:
                print(f"Warning: {search_key} not found. Filling with 0.")
                df[target_col] = 0

        # Fill any tiny holes
        df = df.ffill().bfill() 
        
        # Save to CSV
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/raw_data.csv')
        
        print(f"SUCCESS! Saved {len(df)} rows to data/raw_data.csv")
        
    except Exception as e:
        print(f"Technical Error: {e}")
        import traceback
        traceback.print_exc() # This will show exactly which line failed

if __name__ == "__main__":
    fetch_entsoe_data()