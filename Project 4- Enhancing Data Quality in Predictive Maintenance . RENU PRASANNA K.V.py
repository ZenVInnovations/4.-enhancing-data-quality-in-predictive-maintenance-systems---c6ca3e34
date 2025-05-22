# %% [markdown]
# # Project 4: Enhancing Data Quality in Predictive Maintenance
#
# This notebook demonstrates basic data quality enhancement steps for sensor data using Scikit-learn, focusing on handling missing values and detecting outliers. The data simulates sensor readings from industrial equipment (temperature, pressure, vibration).
#
# **Prerequisite:** Ensure that the necessary libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`) are installed in your Python environment. You should have installed these using `pip install ...` in your terminal or command prompt if running locally, or via your platform's package manager if using an online service.
#
# This notebook will perform the data quality process twice with different random seeds to show two different examples of outputs and graphs.

# %% [markdown]
# ## 1. Import Libraries
#
# We'll start by importing all the necessary libraries for data manipulation, numerical operations, visualization, and machine learning (from scikit-learn).

# %%
# These imports should work if you have installed the libraries correctly
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

print("Libraries imported successfully.")

# %% [markdown]
# # --- Run 1: Using Random Seed 42 ---
#
# This section performs the data quality enhancement and visualization steps on a synthetic dataset generated with `np.random.seed(42)`.

# %% [markdown]
# ## 2. Generate Synthetic Sensor Data (Run 1)
#
# Let's create the first synthetic dataset with missing values and outliers.

# %%
# Set a random seed for reproducibility for Run 1
np.random.seed(42)

# Generate time series data (hourly readings for 1000 hours)
dates_run1 = pd.date_range(start='2023-01-01', periods=1000, freq='h') # Using 'h' for hourly frequency

# Simulate sensor readings with some baseline and trend
temperature_run1 = 25 + 5 * np.random.randn(1000) + np.linspace(0, 10, 1000)
pressure_run1 = 100 + 10 * np.random.randn(1000) - np.linspace(0, 5, 1000)
vibration_run1 = 0.5 + 0.2 * np.random.randn(1000) + 0.1 * np.sin(np.linspace(0, 50, 1000))

# Create a Pandas DataFrame
data_run1 = pd.DataFrame({'timestamp': dates_run1,
                          'temperature': temperature_run1,
                          'pressure': pressure_run1,
                          'vibration': vibration_run1})

# Introduce Missing Values (e.g., 5% missing randomly)
for col in ['temperature', 'pressure', 'vibration']:
    missing_indices = np.random.choice(data_run1.index, size=int(len(data_run1) * 0.05), replace=False)
    data_run1.loc[missing_indices, col] = np.nan

# Introduce Outliers (a few extreme values)
data_run1.loc[150, 'temperature'] = 150
data_run1.loc[600, 'pressure'] = -50
data_run1.loc[850, 'vibration'] = 5
data_run1.loc[200, 'temperature'] = -10


print("Run 1: Synthetic data generated with missing values and outliers.")
print("\nRun 1: Sample Data Head:")
print(data_run1.head())

print("\nRun 1: Missing values before handling:")
print(data_run1.isnull().sum())

# %% [markdown]
# ## 3. Handle Missing Values using Scikit-learn (Run 1)
#
# Imputing missing values for the first dataset.

# %%
numerical_cols = ['temperature', 'pressure', 'vibration']
imputer = SimpleImputer(strategy='median')
data_run1[numerical_cols] = imputer.fit_transform(data_run1[numerical_cols])

print("Run 1: Missing values handled using median imputation.")
print("\nRun 1: Missing values after handling:")
print(data_run1.isnull().sum())

# %% [markdown]
# ## 4. Detect Outliers using Scikit-learn (Run 1)
#
# Detecting outliers using Isolation Forest and LOF for the first dataset.

# %%
# Isolation Forest
iso_forest = IsolationForest(contamination='auto', random_state=42)
data_run1['iso_forest_outlier'] = iso_forest.fit_predict(data_run1[numerical_cols])

# Local Outlier Factor
lof = LocalOutlierFactor(n_neighbors=20)
data_run1['lof_outlier'] = lof.fit_predict(data_run1[numerical_cols])

print("Run 1: Outlier detection completed.")
print("Run 1: Number of outliers detected by Isolation Forest:", data_run1[data_run1['iso_forest_outlier'] == -1].shape[0])
print("Run 1: Number of outliers detected by LOF:", data_run1[data_run1['lof_outlier'] == -1].shape[0])


# %% [markdown]
# ## 5. Visualize Sensor Data and Detected Outliers (Run 1)
#
# Visualizing the data and detected outliers for the first dataset. Includes time series plots and scatter plots highlighting outliers for all variables.

# %% [markdown]
# ### 5.1 Time Series Plots (Run 1)

# %%
plt.figure(figsize=(14, 7))
sns.lineplot(x='timestamp', y='temperature', data=data_run1)
plt.title('Run 1: Temperature Readings Over Time (After Imputation)')
plt.xlabel('Timestamp')
plt.ylabel('Temperature')
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 7))
sns.lineplot(x='timestamp', y='pressure', data=data_run1)
plt.title('Run 1: Pressure Readings Over Time (After Imputation)')
plt.xlabel('Timestamp')
plt.ylabel('Pressure')
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 7))
sns.lineplot(x='timestamp', y='vibration', data=data_run1)
plt.title('Run 1: Vibration Readings Over Time (After Imputation)')
plt.xlabel('Timestamp')
plt.ylabel('Vibration')
plt.grid(True)
plt.show()

# %% [markdown]
# ### 5.2 Scatter Plots with Outliers (Isolation Forest, Run 1)

# %%
plt.figure(figsize=(14, 7))
sns.scatterplot(x='timestamp', y='temperature', hue='iso_forest_outlier', data=data_run1, palette={1: 'blue', -1: 'red'}, s=20)
plt.title('Run 1: Temperature Readings with Isolation Forest Outliers')
plt.xlabel('Timestamp')
plt.ylabel('Temperature')
plt.legend(title='Outlier (-1: Yes, 1: No)')
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 7))
sns.scatterplot(x='timestamp', y='pressure', hue='iso_forest_outlier', data=data_run1, palette={1: 'blue', -1: 'red'}, s=20)
plt.title('Run 1: Pressure Readings with Isolation Forest Outliers')
plt.xlabel('Timestamp')
plt.ylabel('Pressure')
plt.legend(title='Outlier (-1: Yes, 1: No)')
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 7))
sns.scatterplot(x='timestamp', y='vibration', hue='iso_forest_outlier', data=data_run1, palette={1: 'blue', -1: 'red'}, s=20)
plt.title('Run 1: Vibration Readings with Isolation Forest Outliers')
plt.xlabel('Timestamp')
plt.ylabel('Vibration')
plt.legend(title='Outlier (-1: Yes, 1: No)')
plt.grid(True)
plt.show()

# %% [markdown]
# ### 5.3 Scatter Plots with Outliers (Local Outlier Factor, Run 1)

# %%
plt.figure(figsize=(14, 7))
sns.scatterplot(x='timestamp', y='temperature', hue='lof_outlier', data=data_run1, palette={1: 'blue', -1: 'red'}, s=20)
plt.title('Run 1: Temperature Readings with LOF Outliers')
plt.xlabel('Timestamp')
plt.ylabel('Temperature')
plt.legend(title='Outlier (-1: Yes, 1: No)')
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 7))
sns.scatterplot(x='timestamp', y='pressure', hue='lof_outlier', data=data_run1, palette={1: 'blue', -1: 'red'}, s=20)
plt.title('Run 1: Pressure Readings with LOF Outliers')
plt.xlabel('Timestamp')
plt.ylabel('Pressure')
plt.legend(title='Outlier (-1: Yes, 1: No)')
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 7))
sns.scatterplot(x='timestamp', y='vibration', hue='lof_outlier', data=data_run1, palette={1: 'blue', -1: 'red'}, s=20)
plt.title('Run 1: Vibration Readings with LOF Outliers')
plt.xlabel('Timestamp')
plt.ylabel('Vibration')
plt.legend(title='Outlier (-1: Yes, 1: No)')
plt.grid(True)
plt.show()

# %% [markdown]
# ## 6. Handling Detected Outliers (Example: Removal, Run 1)

# %%
data_cleaned_iso_run1 = data_run1[data_run1['iso_forest_outlier'] == 1].copy()

print("\nRun 1: Original data shape:", data_run1.shape)
print("Run 1: Cleaned data shape (after removing Isolation Forest outliers):", data_cleaned_iso_run1.shape)


# %% [markdown]
# # --- Run 2: Using a Different Random Seed (e.g., 99) ---
#
# This section repeats the data quality enhancement and visualization steps on a *new* synthetic dataset generated with a different random seed (`np.random.seed(99)`). This will result in different data, different missing values/outliers, and different graph outputs compared to Run 1.

# %% [markdown]
# ## 2. Generate Synthetic Sensor Data (Run 2)
#
# Generating the second synthetic dataset.

# %%
# Set a DIFFERENT random seed for reproducibility for Run 2
np.random.seed(99)

# Generate time series data (hourly readings for 1000 hours)
dates_run2 = pd.date_range(start='2023-01-01', periods=1000, freq='h') # Using 'h' for hourly frequency

# Simulate sensor readings with the same baseline and trend, but different random noise due to the new seed
temperature_run2 = 25 + 5 * np.random.randn(1000) + np.linspace(0, 10, 1000)
pressure_run2 = 100 + 10 * np.random.randn(1000) - np.linspace(0, 5, 1000)
vibration_run2 = 0.5 + 0.2 * np.random.randn(1000) + 0.1 * np.sin(np.linspace(0, 50, 1000))

# Create a Pandas DataFrame
data_run2 = pd.DataFrame({'timestamp': dates_run2,
                          'temperature': temperature_run2,
                          'pressure': pressure_run2,
                          'vibration': vibration_run2})

# Introduce Missing Values (e.g., 5% missing randomly) - locations will differ due to new seed
for col in ['temperature', 'pressure', 'vibration']:
    missing_indices = np.random.choice(data_run2.index, size=int(len(data_run2) * 0.05), replace=False)
    data_run2.loc[missing_indices, col] = np.nan

# Introduce Outliers (a few extreme values) - locations will be the same as in Run 1, but values might interact differently with random noise
data_run2.loc[150, 'temperature'] = 150
data_run2.loc[600, 'pressure'] = -50
data_run2.loc[850, 'vibration'] = 5
data_run2.loc[200, 'temperature'] = -10


print("\nRun 2: Synthetic data generated with missing values and outliers.")
print("\nRun 2: Sample Data Head:")
print(data_run2.head())

print("\nRun 2: Missing values before handling:")
print(data_run2.isnull().sum())


# %% [markdown]
# ## 3. Handle Missing Values using Scikit-learn (Run 2)
#
# Imputing missing values for the second dataset.

# %%
numerical_cols = ['temperature', 'pressure', 'vibration']
imputer = SimpleImputer(strategy='median')
data_run2[numerical_cols] = imputer.fit_transform(data_run2[numerical_cols])

print("Run 2: Missing values handled using median imputation.")
print("\nRun 2: Missing values after handling:")
print(data_run2.isnull().sum())

# %% [markdown]
# ## 4. Detect Outliers using Scikit-learn (Run 2)
#
# Detecting outliers using Isolation Forest and LOF for the second dataset.

# %%
# Isolation Forest
iso_forest = IsolationForest(contamination='auto', random_state=42) # Using same model parameters
data_run2['iso_forest_outlier'] = iso_forest.fit_predict(data_run2[numerical_cols])

# Local Outlier Factor
lof = LocalOutlierFactor(n_neighbors=20) # Using same model parameters
data_run2['lof_outlier'] = lof.fit_predict(data_run2[numerical_cols])

print("Run 2: Outlier detection completed.")
print("Run 2: Number of outliers detected by Isolation Forest:", data_run2[data_run2['iso_forest_outlier'] == -1].shape[0])
print("Run 2: Number of outliers detected by LOF:", data_run2[data_run2['lof_outlier'] == -1].shape[0])

# %% [markdown]
# ## 5. Visualize Sensor Data and Detected Outliers (Run 2)
#
# Visualizing the data and detected outliers for the second dataset. Includes time series plots and scatter plots highlighting outliers for all variables.

# %% [markdown]
# ### 5.1 Time Series Plots (Run 2)

# %%
plt.figure(figsize=(14, 7))
sns.lineplot(x='timestamp', y='temperature', data=data_run2)
plt.title('Run 2: Temperature Readings Over Time (After Imputation)')
plt.xlabel('Timestamp')
plt.ylabel('Temperature')
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 7))
sns.lineplot(x='timestamp', y='pressure', data=data_run2)
plt.title('Run 2: Pressure Readings Over Time (After Imputation)')
plt.xlabel('Timestamp')
plt.ylabel('Pressure')
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 7))
sns.lineplot(x='timestamp', y='vibration', data=data_run2)
plt.title('Run 2: Vibration Readings Over Time (After Imputation)')
plt.xlabel('Timestamp')
plt.ylabel('Vibration')
plt.grid(True)
plt.show()

# %% [markdown]
# ### 5.2 Scatter Plots with Outliers (Isolation Forest, Run 2)

# %%
plt.figure(figsize=(14, 7))
sns.scatterplot(x='timestamp', y='temperature', hue='iso_forest_outlier', data=data_run2, palette={1: 'blue', -1: 'red'}, s=20)
plt.title('Run 2: Temperature Readings with Isolation Forest Outliers')
plt.xlabel('Timestamp')
plt.ylabel('Temperature')
plt.legend(title='Outlier (-1: Yes, 1: No)')
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 7))
sns.scatterplot(x='timestamp', y='pressure', hue='iso_forest_outlier', data=data_run2, palette={1: 'blue', -1: 'red'}, s=20)
plt.title('Run 2: Pressure Readings with Isolation Forest Outliers')
plt.xlabel('Timestamp')
plt.ylabel('Pressure')
plt.legend(title='Outlier (-1: Yes, 1: No)')
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 7))
sns.scatterplot(x='timestamp', y='vibration', hue='iso_forest_outlier', data=data_run2, palette={1: 'blue', -1: 'red'}, s=20)
plt.title('Run 2: Vibration Readings with Isolation Forest Outliers')
plt.xlabel('Timestamp')
plt.ylabel('Vibration')
plt.legend(title='Outlier (-1: Yes, 1: No)')
plt.grid(True)
plt.show()

# %% [markdown]
# ### 5.3 Scatter Plots with Outliers (Local Outlier Factor, Run 2)

# %%
plt.figure(figsize=(14, 7))
sns.scatterplot(x='timestamp', y='temperature', hue='lof_outlier', data=data_run2, palette={1: 'blue', -1: 'red'}, s=20)
plt.title('Run 2: Temperature Readings with LOF Outliers')
plt.xlabel('Timestamp')
plt.ylabel('Temperature')
plt.legend(title='Outlier (-1: Yes, 1: No)')
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 7))
sns.scatterplot(x='timestamp', y='pressure', hue='lof_outlier', data=data_run2, palette={1: 'blue', -1: 'red'}, s=20)
plt.title('Run 2: Pressure Readings with LOF Outliers')
plt.xlabel('Timestamp')
plt.ylabel('Pressure')
plt.legend(title='Outlier (-1: Yes, 1: No)')
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 7))
sns.scatterplot(x='timestamp', y='vibration', hue='lof_outlier', data=data_run2, palette={1: 'blue', -1: 'red'}, s=20)
plt.title('Run 2: Vibration Readings with LOF Outliers')
plt.xlabel('Timestamp')
plt.ylabel('Vibration')
plt.legend(title='Outlier (-1: Yes, 1: No)')
plt.grid(True)
plt.show()

# %% [markdown]
# ## 6. Handling Detected Outliers (Example: Removal, Run 2)

# %%
data_cleaned_iso_run2 = data_run2[data_run2['iso_forest_outlier'] == 1].copy()

print("\nRun 2: Original data shape:", data_run2.shape)
print("Run 2: Cleaned data shape (after removing Isolation Forest outliers):", data_cleaned_iso_run2.shape)


# %% [markdown]
# ## Conclusion
#
# This notebook demonstrated basic data quality enhancement techniques for sensor data from industrial equipment, showing two different examples generated with different random seeds. The process included handling missing values, detecting outliers using Isolation Forest and Local Outlier Factor, and visualizing the data and results. These steps are fundamental for preparing data for accurate predictive maintenance modeling.

# %%
