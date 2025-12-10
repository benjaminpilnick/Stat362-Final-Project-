# Energy Consumption Prediction Analysis
Stat 362 - Fall 2025
Team Members: Celena Kim, Sarah Kim, Benjamin Pilnick, Emily Yoo


# Problem Description and Dataset 
Residential electricity use comprises a significant portion of overall energy demand, and utilities have to match that demand with supply in real time. When energy usage is hard to predict, it can drive up operating costs, increase reliance on carbon-heavy plants, and put extra strain on the grid. On the household side, having better forecasts of electricity use can help people make smarter choices such as adjusting thermostats, timing when they run appliances, or coordinating things like home batteries and EV charging. Therefore, our goal for this project is to predict hourly household electricity usage (Global_active_power) using past consumption patterns and related electrical measurements. 

The dataset we utilized for this analysis is the "Individual household electric power consumption Data Set" from the UCI Machine Learning Repository. [Link text](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption). The data was collected from a single household over a period of almost 4 years, from December 2006 to November 2010. The dataset contains measurements of electric power consumption in one-minute intervals. We are using Global_active_power as our target variable. 

# High-level description of models
This project includes both baseline models and deep learning architectures. Before running the first model, we preformed the following preprocessing steps:
- Forward-filled missing values  (25k missing values in Sub_metering_3)
- Converted minute-level data → hourly averages
- Parsed timestamps and extracted time features
- Added cyclical time encodings (sin/cos for hour, day, month, weekend)
- Normalized features + target using StandardScaler
- Split dataset into train/validation/test (70%/15%/15%)
- Created sliding window sequences: 24-hour lookback → 1-hour-ahead forecast
- Constructed 3D model inputs

For our baseline model, we utilized a LSTM model that uses the previous 24 hours of energy consumption data to forecast the next hour (one-step-ahead; t+1). We then evaluated the model by generating predictions for every hour in the test set using rolling sliding-window approach (t+1). From this, we received the following results:
Train RMSE: 0.608
Validation RMSE: 0.516
Test RMSE: 0.551
![Baseline curve](Figures/Training:Validation.png)


# Summary of key results
You may reuse plots or metrics from your final presentation slides.

# How to run the code
Basic commands to run one or two representative experiments or demos

