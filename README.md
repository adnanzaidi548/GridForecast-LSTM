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



