# Indian Crop Yield Prediction Using Machine Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning and Exploratory Data Analysis (EDA) project that predicts agricultural crop yields across Indian states by integrating historical crop production records, multi-decade meteorological observations, and state-level soil nutrient profiles.

---

## 📌 Project Overview

Agriculture is the backbone of the Indian economy, employing over 50% of the workforce. However, crop yields are heavily influenced by complex interactions between regional climate patterns, seasonal factors, soil chemical properties, and agricultural inputs (fertilizers and pesticides).

This project integrates three complementary data sources into a unified analytical dataset (19,689 records across 16 features) and trains multiple regression algorithms to accurately forecast crop yields per unit area ($\text{Yield} = \frac{\text{Production}}{\text{Area}}$ in quintals/hectare).

---

## 🚀 Key Features

- **Multi-Source Data Fusion**: Combines historical agricultural yields, state-level climate records (1997–2020), and soil NPK + pH levels into a single pipeline.
- **Target Leakage Prevention**: Intentionally excludes the `production` column from model inputs to prevent target leakage, ensuring predictions rely purely on environmental, geographical, seasonal, and input factors.
- **Extensive Exploratory Data Analysis (EDA)**: Includes 14 dedicated data visualizations analyzing crop yields, spatial distributions, seasonal impacts, weather correlations, and feature relationships.
- **Multi-Model Benchmark**: Trains and benchmarks 5 regression algorithms:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - Support Vector Regressor (SVR)
  - K-Nearest Neighbors Regressor (KNN)
- **Standardized Feature Preprocessing**: Incorporates categorical one-hot encoding, mean imputation, and feature scaling (`StandardScaler`).

---

## 📁 Repository Structure

```text
Indian-Crop-Yield-Prediction/
│
├── Indian Crop Yield Prediction.py     # Main end-to-end Python pipeline (EDA & ML)
├── requirements.txt                    # Project dependencies
├── .gitignore                          # Git ignore rules
├── README.md                           # Comprehensive documentation
│
└── data/                               # Dataset directory
    ├── crop_yield.csv                  # Historical crop yield, input & production data (19,689 rows)
    ├── state_weather_data_1997_2020.csv# State-wise historical weather indicators (721 rows)
    └── state_soil_data.csv             # State-wise soil nutrient levels (N, P, K, pH) (31 rows)
```

---

## 📊 Datasets & Schema

The project utilizes three datasets located in the `data/` directory:

### 1. `crop_yield.csv` (19,689 rows)
Contains agricultural production and farming inputs across various Indian states, crops, and seasons.
- `crop`: Name of the crop (e.g., Rice, Wheat, Sugarcane, Cotton, Maize, Coconut).
- `year`: Harvesting year (1997 to 2020).
- `season`: Cropping season (`Kharif`, `Rabi`, `Whole Year`, `Summer`, `Autumn`, `Winter`).
- `state`: Indian state / union territory.
- `area`: Cultivated area in hectares.
- `production`: Total crop production in metric tonnes/units.
- `fertilizer`: Quantity of fertilizer used (kg).
- `pesticide`: Quantity of pesticide used (kg).
- `yield`: Target variable — calculated yield per unit area.

### 2. `state_weather_data_1997_2020.csv` (721 rows)
Aggregated annual climate indicators across states from 1997 to 2020.
- `state`: Name of the state.
- `year`: Observation year.
- `avg_temp_c`: Mean annual temperature in Celsius (°C).
- `total_rainfall_mm`: Total annual rainfall in millimeters (mm).
- `avg_humidity_percent`: Mean relative humidity percentage (%).

### 3. `state_soil_data.csv` (31 rows)
Baseline soil nutrient parameters for Indian states and union territories.
- `state`: Name of the state.
- `N`: Nitrogen content (kg/ha).
- `P`: Phosphorus content (kg/ha).
- `K`: Potassium content (kg/ha).
- `pH`: Soil acidity / alkalinity index.

---

## 🔬 Data Processing Pipeline

1. **Merging & Cleaning**:
   - `crop` and `weather` datasets are joined on `['state', 'year']` using a left join.
   - The merged dataset is joined with `soil` on `'state'` using a left join.
   - Duplicate rows are removed.
   - Infinite values (`inf`, `-inf`) are replaced with `NaN`.
   - Records with missing target values (`yield`) are dropped.
2. **Feature Engineering & Leakage Prevention**:
   - Features `yield` (target) and `production` (leakage source) are separated from predictors:
     $$X = \text{df.drop}(\text{columns}=[\text{"yield"}, \text{"production"}])$$
   - Categorical variables (`crop`, `season`, `state`) are one-hot encoded using `pd.get_dummies(drop_first=True)`.
   - Any missing numerical values are imputed using column-wise means.
3. **Train-Test Split & Scaling**:
   - Split into 80% training and 20% testing sets with fixed `random_state=42`.
   - Standardized using `StandardScaler` to normalize features for distance- and margin-based models (KNN, SVR).

---

## 📈 Exploratory Data Analysis & Visualizations

The script generates 14 analytical visualizations using `matplotlib` and `seaborn`:

| # | Visualization | Description |
|---|---------------|-------------|
| 1 | **Top 10 Crops by Median Yield** | Log-scaled horizontal bar chart identifying high-yield crops. |
| 2 | **Top 10 Crops by Total Production** | Symmetric log-scaled horizontal bar chart showing major volume crops. |
| 3 | **Top 10 Crops by Cultivated Area** | Bar chart showing the crops covering the largest farming land area. |
| 4 | **Yield Distribution Across Seasons** | Box plot analyzing yield variations across Kharif, Rabi, Summer, Autumn, Winter. |
| 5 | **Rainfall vs. Yield by Crop** | Faceted scatter plots illustrating crop-specific rainfall response. |
| 6 | **Temperature vs. Yield by Crop** | Faceted scatter plots depicting temperature sensitivity per crop. |
| 7 | **Humidity vs. Yield by Crop** | Faceted scatter plots evaluating humidity impact on yield. |
| 8 | **Rainfall vs. Average Temperature** | Multi-season scatter plot examining climatic clustering. |
| 9 | **Median Crop Yield Over Time** | Line plot showing national median yield trends from 1997 to 2020. |
| 10 | **Top 10 States by Median Yield** | Horizontal bar chart ranking the highest-yielding states. |
| 11 | **Top 10 States by Total Production** | State-level agricultural volume ranking. |
| 12 | **State-wise Crop Yield Heatmap** | Pivot matrix showing yield variations of major crops across top states. |
| 13 | **Correlation Matrix Heatmap** | Pearson correlation heatmap across numerical agronomic and weather metrics. |
| 14 | **Actual vs. Predicted Yield (KNN)** | Symmetric log-scale scatter plot comparing predictions against ground truth. |

---

## 🤖 Machine Learning Models & Results

Five regression algorithms were trained and evaluated on the held-out 20% test set (3,938 samples) using three primary metrics:
- **$R^2$ Score (Coefficient of Determination)**: Proportion of variance explained by the model.
- **MAE (Mean Absolute Error)**: Average magnitude of the prediction errors.
- **RMSE (Root Mean Squared Error)**: Standard deviation of residuals, penalizing larger errors.

### Performance Benchmark

| Model | Hyperparameters / Configuration | $R^2$ Score | MAE | RMSE |
| :--- | :--- | :---: | :---: | :---: |
| **K-Nearest Neighbors (KNN)** 🏆 | `n_neighbors=5`, Scaled features | **0.9875** | **7.15** | **99.91** |
| **Decision Tree Regressor** | `min_samples_leaf=2`, `random_state=42` | **0.9717** | **9.51** | **150.62** |
| **Random Forest Regressor** | `n_estimators=100`, `min_samples_leaf=2`, `random_state=42` | **0.9710** | **10.42** | **152.40** |
| **Linear Regression** | Default baseline | **0.8105** | **63.96** | **389.70** |
| **Support Vector Regressor (SVR)** | `kernel='rbf'`, `C=100`, `epsilon=0.1`, Scaled features | **0.7222** | **36.01** | **471.80** |

> **Key Finding**: K-Nearest Neighbors (KNN) and tree-based ensemble methods significantly outperform linear models due to the non-linear interaction between climate, soil, crop type, and regional agronomic conditions.

---

## ⚙️ Installation & Usage

### Prerequisites
- Python 3.8 or higher installed on your system.
- Git installed.

### 1. Clone the Repository
```bash
git clone https://github.com/akshatVerse/Indian-Crop-Yield-Prediction.git
cd Indian-Crop-Yield-Prediction
```

### 2. Create and Activate a Virtual Environment (Optional but Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Pipeline
Execute the main Python script from the project root:
```bash
python "Indian Crop Yield Prediction.py"
```

The script will automatically:
1. Load and merge the datasets from `data/`.
2. Display dataset statistics and missing value checks.
3. Render interactive exploratory visualization windows.
4. Preprocess features and train all 5 regression models.
5. Print performance metrics ($R^2$, MAE, RMSE) and a comparison bar chart.
6. Display a sample table of actual vs. predicted yields.

---

## 📦 Dependencies

The required packages are listed in [`requirements.txt`](requirements.txt):
- `pandas` — Data loading, merging, and tabular transformations
- `numpy` — Mathematical operations and array processing
- `matplotlib` — Base plotting and statistical chart generation
- `seaborn` — Statistical visualizations and heatmaps
- `scikit-learn` — Model training, preprocessing (`StandardScaler`, `train_test_split`), and evaluation metrics

---

## 📄 License & Attribution

- **Source Code**: Released under the [MIT License](LICENSE).
- **Datasets**: Aggregated for educational and academic research from open government and public agronomic datasets (including open records from the Ministry of Agriculture & Farmers Welfare, IMD, and ICAR).

---

## 👤 Author

- GitHub: [@akshatVerse](https://github.com/akshatVerse)
