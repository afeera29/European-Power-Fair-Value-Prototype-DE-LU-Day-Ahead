import pandas as pd
import os
import logging
import json # Added to read the results from the other file
import google.generativeai as genai
from dotenv import load_dotenv

# Setup Logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    filename='logs/ai_errors.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def generate_morning_note(fair_value, market_price, edge, signal):
    """Requirement #4: Automated 'drivers' commentary (Junior Trader Mode)"""
    print("AI is drafting the Junior Trader Morning Note...")
    
    prompt = f"""
    Context: You are a Junior Energy Trader at Cobblestone Energy.
    Our model predicted a Baseload Fair Value of €{fair_value:.2f}/MWh.
    The Market Week-Ahead is trading at €{market_price:.2f}/MWh.
    The calculated Edge is €{edge:.2f}/MWh.
    The suggested Signal is: {signal}.
    
    Task: Write a 3-sentence 'Morning Note' for the Senior Trader. 
    Focus on the fundamental 'Merit Order' logic (e.g., how renewables or load are impacting the fair value).
    Do not invent numbers outside of the ones provided.
    """
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash') # Recommended stable model
        response = model.generate_content(prompt)
        
        with open('output/morning_note.txt', 'w', encoding='utf-8') as f:
            f.write(response.text)
        print("Morning Note drafted successfully.")
        return response.text
    
    except Exception as e:
        logging.error(f"Morning Note Failed: {str(e)}")
        return f"AI Commentary unavailable. Focus on the {edge} edge."

def run_ai_audit(file_path):
    """Existing QA Audit Mode"""
    print("Starting Auditable AI Data Audit...")
    if not os.path.exists(file_path):
        logging.error(f"File {file_path} missing.")
        return

    df = pd.read_csv(file_path)
    stats = {"price_min": float(df['price'].min()), "price_max": float(df['price'].max())}
    
    prompt = f"As an Energy Analyst, identify 2 risks in these German Power stats: {stats}."
    
    with open('logs/last_prompt.txt', 'w') as f:
        f.write(prompt)

    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        with open('output/qa_report.txt', 'w', encoding='utf-8') as f:
            f.write(response.text)
        print("Data Audit complete.")
    except Exception as e:
        logging.error(f"AI Audit Failed: {str(e)}")

if __name__ == "__main__":
    # 1. Running the QA Audit on raw data
    run_ai_audit('data/raw_data.csv')
    
    # DYNAMIC VALUES FROM OTHER FILE
    summary_path = 'output/trading_summary.json'
    
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            data = json.load(f)
        
        # Taking values from the JSON file created by your trading_signal script
        generate_morning_note(
            data['fair_value'], 
            data['market_price'], 
            data['edge'], 
            data['signal']
        )
        print("Morning Note generated using model results from JSON.")
    else:
        # Fallback to the numbers that work if the file isn't there yet
        print("Summary file not found, using fallback values.")
        generate_morning_note(60.07, 95.00, -34.93, "SELL (SHORT) THE WEEK-AHEAD")

    print("\n--- AI AGENT FINISHED ---")