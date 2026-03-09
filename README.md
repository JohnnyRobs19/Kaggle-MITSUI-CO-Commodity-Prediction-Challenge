# MITSUI&CO. Commodity Prediction Challenge

## Competition Summary
The **MITSUI&CO. Commodity Prediction Challenge** challenged participants to develop machine learning models capable of predicting **future commodity price returns** using historical market data from the London Metal Exchange (LME), Japan Exchange Group (JPX), US Stock and Forex markets. The goal was to build stable, long-term forecasts crucial for optimising trading strategies and managing risk in global commodity markets.

---

## Objective
The mission was to predict **price-difference series** — derived from time-series differences between two distinct assets' prices — to extract robust price-movement signals as features and deploy AI-driven trading techniques that turn those signals into sustainable trading profits.

Predictions were made using historical data from four market sources:
- **London Metal Exchange (LME)**
- **Japan Exchange Group (JPX)**
- **US Stock Market**
- **Forex Markets**

---

## Evaluation & Submission
Submissions were made via a **provided evaluation API** that enforces temporal integrity, ensuring models cannot peek forward in time. They are scored using a **Sharpe ratio of daily Spearman rank correlations**:

```
Score = mean(daily_rank_corr) / std(daily_rank_corr)
```

For each date, the Spearman rank correlation is computed between predicted and actual returns across all actively trading securities (nulls from halts, holidays, or delistings are excluded). A higher score reflects both **accuracy** and **consistency** over time.

---

## Impact
This competition advances **AI-driven commodity market forecasting** by:
- Encouraging **robust, adaptable models** that generalise across diverse and evolving market conditions
- Promoting the use of **multi-source financial data** (metals, equities, forex) for richer signal extraction
- Enabling more **stable and accurate long-term price predictions**, reducing financial risk for businesses and investors
- Contributing to more efficient resource allocation and reduced price volatility in global commodity markets

---

## Citation
Maggie Demkin, Naruaki Takano, Rintaro Rai, Sohier Dane, and Tomoya Kitayama. *MITSUI&CO. Commodity Prediction Challenge*. [Kaggle Competition](https://kaggle.com/competitions/mitsui-commodity-prediction-challenge), 2025.
