# German Day-Ahead Power Price Forecasting & Trading Logic

## Project Overview
This project implements an end-to-end pipeline to forecast **German-Luxembourg (DE-LU) Day-Ahead electricity prices**. Using an **XGBoost Regressor** trained on historical ENTSO-E data, the system identifies market "Alpha" by comparing a fundamental-driven **Fair Value** against current Market Forward prices.

## Data Sources & Endpoints
**Provider**: ENTSO-E Transparency Platform.

**Endpoints Used:** > * Day-Ahead Prices [12.1.D]
                        Actual Generation per Production Type [16.1.A] (Wind/Solar)
                        Actual Total Load [6.1.A]

**Access Method:** Programmatic ingestion via the entsoe-py client using REST API endpoints.

### **Core Objectives Fulfilled:**
* **Data Engineering:** Automated ENTSO-E ingestion with DST-aware cleaning and UTC synchronization.
* **QA Auditing:** Systematic reporting on data missingness, coverage, and outliers.
* **Machine Learning:** Gradient Boosting model with a **51.8% performance uplift** over the seasonal naive baseline.
* **Trading Logic:** Quantitative Edge calculation with confidence-weighted signaling.
* **AI Multiplier:** Programmatic LLM integration for automated daily market driver commentary and data risk auditing.


## System Architecture
The project is structured into modular components to ensure production-readiness:

CobblestoneEnergy_Project/
├── data/               # CSV datasets
│   ├── raw_data.csv    # Original API pull
│   └── processed_data.csv # Engineered features
├── src/                # Functional Python logic
│   ├── data_fetcher.py # ENTSO-E API Client
│   ├── cleaning.py     # DST handling, QA checks, & Feature Engineering
│   ├── model.py        # XGBoost training & Walk-forward Validation
│   ├── ai_agent.py     # Programmatic LLM Market Commentary 
│   └── trading_signal.py # Edge calculation & Signal Generation
├── output/                     # Analytics & Trading Deliverables
│   ├── submission.csv          # Final predicted curve 
│   ├── final_forecasts.csv     # Detailed backtest results for trading logic
│   ├── feature_importance_audit.csv # Numerical breakdown of model drivers
│   ├── model_comparison.png    # Visual forecast performance (Actual vs. Predicted)
│   ├── shap_summary.png        # Interpretability Plot (Merit Order impact)
│   ├── morning_note.txt        # AI-Generated Market Briefing 
│   ├── qa_report_processed.txt  # Post-cleaning data integrity report
│   ├── qa_report_raw.txt       # Initial data ingestion audit
│   ├── qa_report.txt           # General QA summary
│   └── risk_audit.txt          # Tail-risk and error distribution analysis       
├── logs/               # AI Audit logs & Error tracking
└── README.md           # Project Documentation


## Data Engineering & QA
Power data is dirty due to sensor failures and Daylight Savings jumps. This pipeline implements:
1.  **The Feature Set:** * **Residual Load:** `Actual Load - (Wind + Solar)`. Captures the **Merit Order Effect**, determining the marginal plant.
    * **Solar Peak Impact:** An interaction feature (`Solar * Hour_Cos`) to model the "Duck Curve" suppression of midday prices.
    * **Autocorrelation Lags:** 24h and 168h (weekly) lags to capture mean-reverting price behaviors.
2.  **Automated QA Audit:** Every run generates a `qa_report_processed.txt`. The cleaning process successfully resolved a **234% raw coverage error** (due to API overlaps) into a perfect **100% hourly timeline**.


## Model Performance & Interpretability
The model was validated using an **Expanding Window Walk-Forward** approach (5 Folds), mimicking a real-world trading environment.

| Metric | Result | Interpretation |
| **Final MAE** | **€15.11/MWh** | High precision in the volatile DE-LU regime. |
| **Performance Uplift** | **50.4%** | Over 2x more accurate than the Seasonal Naive baseline. |
| **Tail Risk (P95)** | **€43.49/MWh** | Quantifies risk during extreme scarcity events. |

## SHAP & Auditability
**SHAP Interpretation**: Using SHAP, it is proven that Residual Load is the primary price driver. High values (low renewables) correlate with positive price pressure, confirming the model has learned the physical constraints of the grid.

**Feature Audit** : The feature_importance_audit.csv provides a mathematical rank of every driver, allowing traders to verify the "weight" of renewables vs. load in every prediction.

## AI-Accelerated Workflow (Requirement #4)
A **Programmatic AI Agent** (`src/ai_agent.py`) was implemented that acts as a "Junior Trader."
* **Engineering Multiplier:** Instead of manual analysis, the AI is called via the **Gemini API** to generate a "Morning Note."
* **Grounding:** The LLM is strictly fed the computed metrics (Residual Load, Edge, Fair Value) to prevent hallucinations.
* **Auditability:** Every prompt and response is logged in `logs/` for transparency and risk management.



## Trading Strategy & Curve Translation
**The "Fair Value" View:**
The model produces 24 hourly forecasts, averaged to create a **Baseload Fair Value**.
* **Signal Logic:** A trade is only triggered if the `Edge > Confidence Threshold`.
* **Desk Action:** If Fair Value is significantly below Market Forward, the desk initiates a **Short Position** in the Prompt (W+1) curve.
* **Invalidation:** Signals are invalidated if actual wind generation deviates from the forecast by >15%.

### How to Run the Pipeline
To reproduce the results and generate the trading signals, execute the scripts in the following order:

1. **Configure Environment**: Create a .env file in the root directory and add your keys:
   ENTSOE_API_KEY=your_key_here
   GEMINI_API_KEY=your_key_here

2. **Data Ingestion**: python src/data_fetcher.py
   (Downloads raw market data to data/raw_data.csv)

3. **Data Cleaning & QA**: python src/cleaning.py
   (Fixes gaps/DST, engineers features, and generates QA reports in output/)

4. **Model Training & Evaluation**: python src/model.py
   (Trains XGBoost, saves the submission.csv, and generates SHAP interpretability plots)

5. **Trading Execution Logic**: python src/trading_signal.py
   (Calculates the Edge and determines the Buy/Sell/Neutral signal)

6. **AI Briefing Generation**: python src/ai_agent.py
   (Calls the Gemini API to draft the final morning_note.txt based on the pipeline's output)