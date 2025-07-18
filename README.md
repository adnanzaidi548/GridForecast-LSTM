# GridForecast-LSTM
GridForecast-LSTM introduces a high-precision, utility-oriented voltage forecasting framework tailored for hourahead predictions in smart distribution systems. By integrating historical voltage data, environmental parameters, and load variability into a Long Short-Term Memory (LSTM) architecture.

README GridForecast-LSTM


GridForecast-LSTM: A Utility-Centric Model for
Hour-Ahead Voltage Forecasting
Adnan Haider Zaidi, Member, IEEE


GitHub Repository (Mandatory Link)
•	Repository URL:   https://github.com/adnanzaidi548/GridForecast-LSTM


GridForecast-LSTM introduces a high-precision,
utility-oriented voltage forecasting framework tailored for hourahead
predictions in smart distribution systems. By integrating
historical voltage data, environmental parameters, and load
variability into a Long Short-Term Memory (LSTM) architecture,
this model achieves dynamic adaptability and robust
temporal learning. The novelty lies in its ability to self-tune
under grid state fluctuations, accommodating distributed energy
resources (DERs), electric vehicle load surges, and weatherinduced
instability. Addressing the critical need for predictive
voltage control, GridForecast-LSTM enhances grid resilience,
minimizes transformer stress, and reduces blackout probabilities.
Future applications include real-time integration into SCADA
and digital twin environments for live utility feedback loops

Source of the Dataset
•	Primary Dataset:
Publicly available from the Ontario Independent Electricity System Operator (IESO) Smart Metering Entity (SME) repository.
•	Time Resolution:
Hourly data over one year (i.e., 8760 data points).
•	Granularity:
Residential substations and distribution feeders.

Each input vector at time t, denoted xt ∈ ℝ⁵, includes:
•	Vt: Voltage [Volts]
•	Pt: Load [kW]
•	Tt: Temperature [°C]
•	Ht: Humidity [%]
•	Wt: Wind speed [km/h]


A. Cleaning & Noise Removal
•	Imputation for missing values via:
o	Linear interpolation
o	Temporal K-Nearest Neighbors (KNN)
•	Outlier detection using IQR thresholds or Z-score method
B. Normalization
•	Min-Max Scaling across all features:
X′=X−XminXmax−XminX' = \frac{X - X_{\text{min}}}{X_{\text{max}} - X_{\text{min}}} 
C. Temporal Structuring
•	Sliding Window with:
o	Input sequence length (lookback) = 24 hours
o	Prediction horizon = 1 hour ahead
________________________________________
🧠 4. Target Output
•	Forecasted voltage value (V̂t+1) at the feeder level, 1 hour ahead.
________________________________________
🛠️ 5. Jupyter Notebook Code Requirements
Modules to be included:
•	pandas, numpy, matplotlib, scikit-learn for preprocessing & visualization
•	tensorflow and keras for LSTM model training
•	optuna or kerastuner for hyperparameter tuning
•	shap or lime for interpretability
Notebook Sections:
Step	Notebook Cell Purpose
1.	Data loading from IESO CSV / URL
2.	Data cleaning, imputation, noise filtering
3.	Feature engineering (Voltage, Load, Temp, etc.)
4.	Sliding window transformation
5.	Train/test/validation split (70/15/15)
6.	Min-max normalization
7.	LSTM model definition (2 or 3 hidden layers, dropout, attention)
8.	Model training and loss tracking
9.	Performance evaluation: MAE, RMSE, MAPE, R²
10.	Comparison plots: predicted vs actual voltage
11.	Export model checkpoints and predictions
12.	GitHub integration and documentation for reproducibility
________________________________________
📈 6. Experimental Setup Summary
•	Hardware: NVIDIA GPU (e.g., Tesla T4 / RTX 3090 / A100)
•	Software: Python 3.10 / 3.11, TensorFlow 2.12/2.14
•	Batch Size: 64
•	Epochs: 100–200
•	Optimizer: Adam (lr=0.001)
•	Loss Function: MSE or MAE
•	Dropout: 0.2–0.3
•	Validation Split: 15%

GitHub Repository (Mandatory Link)
•	Repository URL:


https://github.com/adnanzaidi548/GridForecast-LSTM


•	Shape of X_train, y_train, X_val, X_test
•	Example input window: [[Vt, Tt, Ht, Wt, Pt] for 24 timesteps]
•	Visualizations:
o	Actual vs Predicted Voltage Plot (24–48 hour span)
o	SHAP summary plot (feature importance)
o	Training loss & validation loss curves
•	Final results (e.g., MAE = 0.96, RMSE = 1.12, MAPE = 1.83%)





 
GridForecast-LSTM: A Utility-Centric Model for
Hour-Ahead Voltage Forecasting
Adnan Haider Zaidi, Member, IEEE
STEPS TAKEN FOR DATA PROCESSING AND PYTHON CODES

 
Abstract—GridForecast-LSTM introduces a high-precision, utility-oriented voltage forecasting framework tailored for hourahead predictions in smart distribution systems. By integrating
historical voltage data, environmental parameters, and load variability into a Long Short-Term Memory (LSTM) architecture, this model achieves dynamic adaptability and robust
temporal learning. The novelty lies in its ability to self-tune under grid state fluctuations, accommodating distributed energy resources (DERs), electric vehicle load surges, and weatherinduced instability. Addressing the critical need for predictive voltage control, GridForecast-LSTM enhances grid resilience, minimizes transformer stress, and reduces blackout probabilities. Future applications include real-time integration into SCADA and digital twin environments for live utility feedback loops.

Index Terms—Voltage  Forecasting, LSTM, smart grids, realtime prediction, utility analytics, DER integration, predictive control, digital twin.

Title: Data Processing and Modeling Pipeline for GridForecast-LSTM (IEEE Smart Grid)

This document outlines the complete step-by-step process used in Google Colab for the GridForecast-LSTM model implementation using the dataset hosted at:
https://github.com/adnanzaidi548/GridForecast-LSTM/blob/main/GridForecast_LSTM_Synthetic.csv

---

 Step-by-Step Pipeline Overview 

1. **Load Required Libraries**
   - Import `pandas`, `numpy`, `scikit-learn`, `matplotlib`, etc.
   - These libraries support data processing, visualization, and modeling.

2. **Load Dataset from GitHub**
   - Load CSV using the raw GitHub URL.
   - Parse timestamps and preview the dataset.

3. **Handle Missing Values**
   - Apply linear interpolation for missing data.
   - Ensure no NaNs are left in key input columns.

4. **Outlier Detection and Removal**
   - Use Interquartile Range (IQR) to remove statistical outliers.
   - Filter values outside 1.5*IQR bounds for each feature.

5. **Min-Max Normalization**
   - Normalize all numerical features to the range [0, 1] using `MinMaxScaler`.
   - Features: Voltage, Load, Temperature, Humidity, Wind Speed.

6. **Sliding Window Transformation**
   - Create supervised learning sequences using a 24-hour lookback window.
   - Prediction target: Voltage at t+1 hour.

7. **Train/Validation/Test Split**
   - Divide data into 70% Train, 15% Validation, 15% Test sets.
   - Maintain temporal order (no shuffling).

8. **Define LSTM Model Architecture**
   - Build a sequential LSTM model with 2–3 hidden layers.
   - Include dropout layers to prevent overfitting.

9. **Compile and Train the Model**
   - Use `Adam` optimizer with learning rate = 0.001.
   - Loss Function: Mean Squared Error (MSE).
   - Train for 100–200 epochs with early stopping.

10. **Evaluate Model Performance**
    - Calculate MAE, RMSE, MAPE, and R².
    - Visualize actual vs. predicted voltage values.

11. **Interpretability with SHAP**
    - Use SHAP values to analyze feature importance (optional).

12. **Export Trained Model**
    - Save model as `.h5` or `.pkl` format.
    - Save normalized scalers and metadata if needed.

13. **GitHub Integration (Optional)**
    - Push training logs, final model, and notebook to GitHub repository.

14. **Documentation and Reproducibility**
    - Include README, installation guide, and reproducibility instructions.
    - Document model architecture and input/output formats.

---

This plan aligns with IEEE Smart Grid reproducibility requirements and utility-grade deployment scenarios.________________________________________
 1. Source of the Dataset
•	Primary Dataset:
Publicly available from the Ontario Independent Electricity System Operator (IESO) Smart Metering Entity (SME) repository.
•	Time Resolution:
Hourly data over one year (i.e., 8760 data points).
•	Granularity:
Residential substations and distribution feeders.
________________________________________
 2. Data Features (Variables)
Each input vector at time t, denoted xt ∈ ℝ⁵, includes:
•	Vt: Voltage [Volts]
•	Pt: Load [kW]
•	Tt: Temperature [°C]
•	Ht: Humidity [%]
•	Wt: Wind speed [km/h]
________________________________________

 3. Preprocessing Pipeline
All steps must be implemented in the notebook:
A. Cleaning & Noise Removal
•	Imputation for missing values via:
o	Linear interpolation
o	Temporal K-Nearest Neighbors (KNN)
•	Outlier detection using IQR thresholds or Z-score method
B. Normalization
•	Min-Max Scaling across all features:
X′=X−XminXmax−XminX' = \frac{X - X_{\text{min}}}{X_{\text{max}} - X_{\text{min}}} 
C. Temporal Structuring
•	Sliding Window with:
o	Input sequence length (lookback) = 24 hours
o	Prediction horizon = 1 hour ahead
________________________________________
 4. Target Output
•	Forecasted voltage value (V̂t+1) at the feeder level, 1 hour ahead.
________________________________________
 5. Jupyter Notebook Code Requirements
Modules to be included:
•	pandas, numpy, matplotlib, scikit-learn for preprocessing & visualization
•	tensorflow and keras for LSTM model training
•	optuna or kerastuner for hyperparameter tuning
•	shap or lime for interpretability
Notebook Sections:
Step	Notebook Cell Purpose
1.	Data loading from IESO CSV / URL
2.	Data cleaning, imputation, noise filtering
3.	Feature engineering (Voltage, Load, Temp, etc.)
4.	Sliding window transformation
5.	Train/test/validation split (70/15/15)
6.	Min-max normalization
7.	LSTM model definition (2 or 3 hidden layers, dropout, attention)
8.	Model training and loss tracking
9.	Performance evaluation: MAE, RMSE, MAPE, R²
10.	Comparison plots: predicted vs actual voltage
11.	Export model checkpoints and predictions
12.	GitHub integration and documentation for reproducibility
________________________________________


 6. Experimental Setup Summary
•	Hardware: NVIDIA GPU (e.g., Tesla T4 / RTX 3090 / A100)
•	Software: Python 3.10 / 3.11, TensorFlow 2.12/2.14
•	Batch Size: 64
•	Epochs: 100–200
•	Optimizer: Adam (lr=0.001)
•	Loss Function: MSE or MAE
•	Dropout: 0.2–0.3
•	Validation Split: 15%
________________________________________
 7. GitHub Repository (Mandatory Link)
•	Repository URL:
https://github.com/adnanhaiderzaidi/GridForecast-LSTM
________________________________________
 Notebook Data Output
•	Shape of X_train, y_train, X_val, X_test
•	Example input window: [[Vt, Tt, Ht, Wt, Pt] for 24 timesteps]
•	Visualizations:
o	Actual vs Predicted Voltage Plot (24–48 hour span)
o	SHAP summary plot (feature importance)
o	Training loss & validation loss curves
•	Final results (e.g., MAE = 0.96, RMSE = 1.12, MAPE = 1.83%)
________________________________________
Our data link at Github:
https://github.com/adnanzaidi548/GridForecast-LSTM/blob/main/GridForecast_LSTM_Synthetic.csv

This complete Jupyter Notebook code will:
1.	Load the CSV file from GitHub
2.	Clean the data
3.	Remove outliers
4.	Normalize with Min-Max scaling
5.	Prepare data using a sliding window
6.	Split into Train/Val/Test




