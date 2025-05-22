Python 3.13.2 (tags/v3.13.2:4f8bb39, Feb  4 2025, 15:23:48) [MSC v.1942 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> 
= RESTART: E:/Project 4- Enhancing Data Quality in Predictive Maintenance . RENU PRASANNA K.V.py
Libraries imported successfully.
Run 1: Synthetic data generated with missing values and outliers.

Run 1: Sample Data Head:
            timestamp  temperature    pressure  vibration
0 2023-01-01 00:00:00    27.483571  113.993554   0.364964
1 2023-01-01 01:00:00    24.318689  109.241332   0.476099
2 2023-01-01 02:00:00    28.258463  100.586294   0.351509
3 2023-01-01 03:00:00    32.645179   93.515617   0.453366
4 2023-01-01 04:00:00    23.869273  106.962213   0.141164

Run 1: Missing values before handling:
timestamp       0
temperature    49
pressure       50
vibration      50
dtype: int64
Run 1: Missing values handled using median imputation.

Run 1: Missing values after handling:
timestamp      0
temperature    0
pressure       0
vibration      0
dtype: int64
Run 1: Outlier detection completed.
Run 1: Number of outliers detected by Isolation Forest: 136
Run 1: Number of outliers detected by LOF: 24
