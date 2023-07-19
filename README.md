# TXF Day-Trading Machine Learning Stategy
## Strategy Introduction
1. **Data**: TXF market price/volume data with 5mins frequency  
2. **Model Input**: price/volume features (including features after feature interaction)  
3. **Mode**: LSTM model with time windows = 10  
4. **Model Output**: a float between 0 and 1  
5. **Trading Signal**:
- mean = mean of all model output
- stdev = standard deviation of all model output
- LONG! - if current model output > mean + stdev 
- SHORT! - if current model output < mean - stdev
## Files Object
- `strategy_backtesting.ipynb`: model training and strategy backtesting, built with python
- `strategy_production.py`: real market trading code using shioaji API, built with python
- `transactions_0605_to_0616.csv`: real market trading results with 11.3% returns within 2 weeks
- others: data files for model training and backtesting