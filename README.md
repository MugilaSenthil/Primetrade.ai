# Crypto Trader Sentiment Analysis

This project analyzes the relationship between the Bitcoin Fear & Greed Index and trader‑level performance, and builds models to predict whether a trader’s day will be profitable based on sentiment and trading behavior.

## Project Structure
<code>
.
├── csv_files/
│   ├── fear_greed_index.csv               # Raw sentiment data
│   ├── historical_data.csv                # Raw trader-level trade history
│   ├── trader_daily_agg.csv               # Per-account, per-day features
│   ├── top_trader_daily.csv               # Filtered subset: consistent profitable traders
│   ├── executive_summary.csv              # Aggregated metrics by sentiment regime
│   └── model_predictions.csv              # Test-set predictions from ML models
│
├── models/
│   ├── lr_profitable_day.pkl              # Logistic Regression model
│   ├── xgb_profitable_day.pkl             # XGBoost-style Gradient Boosting model
│   └── scaler.pkl                         # Feature scaler used during training
│
├── outputs/
│   ├── sentiment_over_time.png
│   ├── performance_by_sentiment.png
│   ├── avg_daily_pnl_by_sentiment.png
│   ├── avg_leverage_by_sentiment.png
│   ├── win_rate_by_sentiment.png
│   ├── xgb_feature_importance.png
│   └── model_insights.png
│
├── notebook_1.ipynb                       # EDA + feature engineering
├── notebook_2.ipynb                       # Modeling and evaluation
├── ds_report.pdf                          # Final report (assignment deliverable)
└── README.md
</code>

## Objective

- Explore trader‑level performance in the context of market sentiment.
- Engineer daily features that combine trading metrics and sentiment scores.
- Train and evaluate models to predict if a trader’s day will be profitable (`profitable_day` = 1/0).
- Summarize findings and practical implications for trading strategies.


## Data

- **Fear & Greed Index (`fear_greed_index.csv`)**  
  Daily sentiment score for the Bitcoin market, including numeric value and textual classification (e.g., *Extreme Fear*, *Fear*, *Neutral*, *Greed*, *Extreme Greed*).
- **Trader History (`historical_data.csv`)**  
  On‑chain trade‑level history with account, symbol, execution price, size in tokens/USD, side, timestamps, fees, and realized PnL.

---

## Notebooks

### `notebook_1.ipynb` – EDA & Feature Engineering

Main steps:

1. Load both raw datasets and inspect shapes and schemas.
2. Clean Fear & Greed data:  
   - Convert timestamps to dates.  
   - Map sentiment text to an ordinal `sentiment_numeric` scale.
3. Clean trader data:  
   - Standardize column names, parse timestamps, derive `date`.  
   - Filter invalid records and remove extreme PnL outliers.
4. Merge trader data with daily sentiment on `date`.
5. Aggregate to daily per‑account level to create:
   - `trades_count`, `net_pnl`, `avg_pnl`, `total_volume`, `profitable_trades`, `win_rate`, and `profitable_day`.
6. Save processed datasets:
   - `trader_daily_agg.csv`
   - `top_trader_daily.csv` (optional filter for consistently profitable traders)
   - `executive_summary.csv` (summary metrics by sentiment classification)

### `notebook_2.ipynb` – Modeling & Evaluation

Main steps:

1. Load `trader_daily_agg.csv`.
2. Define target: `profitable_day` (1 if daily net PnL > 0, else 0).
3. Select features:
   - `sentiment_numeric`, `value`
   - `trades_count`, `avg_pnl`, `win_rate`, `total_volume`
4. Handle missing sentiment and scale features.
5. Train/test split with stratification.
6. Train models:
   - Logistic Regression (baseline, interpretable).
   - GradientBoostingClassifier (non‑linear “XGBoost‑style” model).
7. Evaluate with cross‑validated and test AUC, confusion matrices, and feature importance.
8. Save artifacts:
   - `models/lr_profitable_day.pkl`
   - `models/xgb_profitable_day.pkl`
   - `models/scaler.pkl`
   - `csv_files/model_predictions.csv`


## How to Run

1. Install dependencies (example with `pip`):

pip install pandas numpy scikit-learn seaborn matplotlib joblib


2. Open `notebook_1.ipynb` and run all cells to:
- Clean and merge data.
- Generate feature datasets and EDA plots.

3. Open `notebook_2.ipynb` and run all cells to:
- Train models.
- Evaluate performance.
- Export model binaries and prediction files.

All paths assume the repository is run from the project root.


## Results (High Level)

- Profitable days can be predicted with AUC above 0.7 using a combination of sentiment and trader behavior features.
- Fear‑related regimes tend to coincide with higher win rates for skilled traders, suggesting a contrarian edge relative to aggregate sentiment.
- Volume, win rate, and sentiment level are among the most important predictors in the gradient boosting model, according to feature importance.


## Future Work

- Add hyperparameter tuning and calibration plots.
- Incorporate additional on‑chain or order book features.

- Package the best model behind a simple API or interactive dashboard.




