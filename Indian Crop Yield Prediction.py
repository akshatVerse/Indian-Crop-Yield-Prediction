
# ============================================================
# INDIAN CROP YIELD PREDICTION
# Data Analysis, Visualization and Machine Learning
# ============================================================


# 1. IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


sns.set_theme(style="whitegrid")


# ============================================================
# 2. LOAD THE DATASETS
# ============================================================

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."

crop = pd.read_csv(
    os.path.join(BASE_DIR, "data", "crop_yield.csv")
)

weather = pd.read_csv(
    os.path.join(BASE_DIR, "data", "state_weather_data_1997_2020.csv")
)

soil = pd.read_csv(
    os.path.join(BASE_DIR, "data", "state_soil_data.csv")
)


# ============================================================
# 3. COMBINE THE DATASETS
# ============================================================

df = pd.merge(crop, weather, on=["state", "year"], how="left")

df = pd.merge(df, soil, on="state", how="left")

df = df.drop_duplicates()

df = df.replace([np.inf, -np.inf], np.nan)

df = df.dropna(subset=["yield"])

print("Dataset shape:", df.shape)

print("\nFirst five rows:")
print(df.head())

print("\nMissing values in each column:")
print(df.isnull().sum())


# ============================================================
# 4. BASIC DATA UNDERSTANDING
# ============================================================

print("\nDataset information:")
df.info()

print("\nStatistical summary:")
print(df.describe())

print("\nNumber of different crops:", df["crop"].nunique())

print("Number of states:", df["state"].nunique())

print("Available seasons:")
print(df["season"].unique())


# ============================================================
# 5. VISUALIZATION 1: YIELD COMPARISON ACROSS CROPS
# ============================================================

# VISUALIZATION 1: TOP 10 CROPS BY MEDIAN YIELD

crop_yield = df.groupby("crop")["yield"].median()

top_yield_crops = crop_yield.sort_values(
    ascending=False
).head(10)

plt.figure(figsize=(12, 6))

plt.barh(
    top_yield_crops.sort_values().index,
    top_yield_crops.sort_values().values,
    color="cornflowerblue"
)

plt.xscale("log")
plt.title("Top 10 Crops by Median Yield")
plt.xlabel("Median Yield (Log Scale)")
plt.ylabel("Crop")

plt.tight_layout()
plt.show()

# ============================================================
# 6. VISUALIZATION 2: TOTAL PRODUCTION BY CROP
# ============================================================

crop_production = df.groupby("crop")["production"].sum()

top_production_crops = crop_production.sort_values(
    ascending=False
).head(10)

plt.figure(figsize=(12, 7))

plt.barh(
    top_production_crops.sort_values().index,
    top_production_crops.sort_values().values,
    color="seagreen"
)

plt.xscale("symlog", linthresh=1)
plt.title("Top 10 Crops by Total Production")
plt.xlabel("Total Production (Symmetric Log Scale)")
plt.ylabel("Crop")
plt.tight_layout()
plt.show()



# ============================================================
# 7. VISUALIZATION 3: CULTIVATED AREA BY CROP
# ============================================================

crop_area = df.groupby("crop")["area"].sum()

top_area_crops = crop_area.sort_values(ascending=False).head(10)

plt.figure(figsize=(12, 6))
top_area_crops.sort_values().plot(kind="barh")

plt.title("Top 10 Crops by Total Cultivated Area")
plt.xlabel("Total Area")
plt.ylabel("Crop")
plt.tight_layout()
plt.show()


# ============================================================
# 8. VISUALIZATION 4: YIELD ACROSS SEASONS
# ============================================================

plt.figure(figsize=(11, 6))

sns.boxplot(
    data=df,
    x="season",
    y="yield",
    showfliers=False
)

plt.title("Crop Yield Distribution Across Seasons")
plt.xlabel("Season")
plt.ylabel("Yield")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


# ============================================================
# 9. SELECT COMMON CROPS FOR WEATHER COMPARISONS
# ============================================================

common_crops = df["crop"].value_counts().head(5).index

crop_weather = df[df["crop"].isin(common_crops)].copy()


# ============================================================
# 10. VISUALIZATION 5: RAINFALL VS YIELD BY CROP
# ============================================================

sns.relplot(
    data=crop_weather,
    x="total_rainfall_mm",
    y="yield",
    col="crop",
    col_wrap=3,
    kind="scatter",
    height=4,
    facet_kws={"sharey": False}
)

plt.suptitle("Rainfall vs Yield Across Different Crops", y=1.03)
plt.show()


# ============================================================
# 11. VISUALIZATION 6: TEMPERATURE VS YIELD BY CROP
# ============================================================

sns.relplot(
    data=crop_weather,
    x="avg_temp_c",
    y="yield",
    col="crop",
    col_wrap=3,
    kind="scatter",
    height=4,
    facet_kws={"sharey": False}
)

plt.suptitle("Temperature vs Yield Across Different Crops", y=1.03)
plt.show()


# ============================================================
# 12. VISUALIZATION 7: HUMIDITY VS YIELD BY CROP
# ============================================================

sns.relplot(
    data=crop_weather,
    x="avg_humidity_percent",
    y="yield",
    col="crop",
    col_wrap=3,
    kind="scatter",
    height=4,
    facet_kws={"sharey": False}
)

plt.suptitle("Humidity vs Yield Across Different Crops", y=1.03)
plt.show()


# ============================================================
# 13. VISUALIZATION 8: RAINFALL VS TEMPERATURE
# ============================================================

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="avg_temp_c",
    y="total_rainfall_mm",
    hue="season",
    alpha=0.6
)

plt.title("Rainfall vs Average Temperature")
plt.xlabel("Average Temperature (°C)")
plt.ylabel("Total Rainfall (mm)")
plt.tight_layout()
plt.show()


# ============================================================
# 14. VISUALIZATION 9: CROP YIELD OVER THE YEARS
# ============================================================

yearly_yield = df.groupby("year")["yield"].median()

plt.figure(figsize=(12, 6))

yearly_yield.plot(kind="line", marker="o")

plt.title("Median Crop Yield Over the Years")
plt.xlabel("Year")
plt.ylabel("Median Yield")
plt.tight_layout()
plt.show()


# ============================================================
# 15. VISUALIZATION 10: YIELD COMPARISON BY STATE
# ============================================================

state_yield = df.groupby("state")["yield"].median()

top_yield_states = state_yield.sort_values(ascending=False).head(10)

plt.figure(figsize=(12, 6))

top_yield_states.sort_values().plot(kind="barh")

plt.title("Top 10 States by Median Crop Yield")
plt.xlabel("Median Yield")
plt.ylabel("State")
plt.tight_layout()
plt.show()


# ============================================================
# 16. VISUALIZATION 11: TOTAL PRODUCTION BY STATE
# ============================================================

state_production = df.groupby("state")["production"].sum()

top_production_states = state_production.sort_values(
    ascending=False
).head(10)

plt.figure(figsize=(12, 6))

top_production_states.sort_values().plot(kind="barh")

plt.title("Top 10 States by Total Crop Production")
plt.xlabel("Total Production")
plt.ylabel("State")
plt.tight_layout()
plt.show()


# ============================================================
# 17. VISUALIZATION 12: STATE-WISE CROP COMPARISON
# ============================================================

common_crops = df["crop"].value_counts().head(5).index

common_states = df["state"].value_counts().head(8).index

state_crop_data = df[
    df["crop"].isin(common_crops)
    & df["state"].isin(common_states)
]

state_crop_table = state_crop_data.pivot_table(
    index="state",
    columns="crop",
    values="yield",
    aggfunc="median"
)

plt.figure(figsize=(12, 7))

sns.heatmap(
    state_crop_table,
    annot=True,
    fmt=".1f",
    cmap="YlGnBu"
)

plt.title("Median Yield of Selected Crops Across States")
plt.xlabel("Crop")
plt.ylabel("State")
plt.tight_layout()
plt.show()


# ============================================================
# 18. VISUALIZATION 13: CORRELATION HEATMAP
# ============================================================

correlation_columns = [
    "yield",
    "area",
    "production",
    "year",
    "avg_temp_c",
    "total_rainfall_mm",
    "avg_humidity_percent"
]

correlation_data = df[correlation_columns].corr()

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation_data,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    vmin=-1,
    vmax=1
)

plt.title("Correlation Between Selected Numerical Features")
plt.tight_layout()
plt.show()


# ============================================================
# 19. PREPARE DATA FOR MACHINE LEARNING
# ============================================================

# Production is excluded to avoid target leakage.

X = df.drop(columns=["yield", "production"])

y = df["yield"]

# Convert crop, season and state into numerical columns.

X = pd.get_dummies(
    X,
    columns=["crop", "season", "state"],
    drop_first=True
)

# Replace any remaining missing numeric values with column means.

X = X.fillna(X.mean())

print("\nFeatures used for prediction:")
print(X.columns.tolist())

print("\nTarget variable: yield")

print("Input data shape:", X.shape)


# ============================================================
# 20. SPLIT THE DATA INTO TRAINING AND TESTING SETS
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining records:", X_train.shape[0])

print("Testing records:", X_test.shape[0])

# Scaling

sc = StandardScaler()

X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)


# ============================================================
# 21. TRAIN THREE MACHINE LEARNING MODELS
# ============================================================

# Model 1: Linear Regression

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

linear_predictions = linear_model.predict(X_test)


# Model 2: Decision Tree

tree_model = DecisionTreeRegressor(
    random_state=42,
    min_samples_leaf=2
)

tree_model.fit(X_train, y_train)

tree_predictions = tree_model.predict(X_test)


# Model 3: Random Forest

forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=2
)

forest_model.fit(X_train, y_train)

forest_predictions = forest_model.predict(X_test)


# ============================================================
# 22. COMPARE MODEL PERFORMANCE
# ============================================================



# 1. Linear Regression
print("\nLinear Regression")
print("R2 Score:", r2_score(y_test, linear_predictions))
print("MAE:", mean_absolute_error(y_test, linear_predictions))
print("RMSE:", np.sqrt(mean_squared_error(y_test, linear_predictions)))


# 2. Decision Tree
print("\nDecision Tree")
print("R2 Score:", r2_score(y_test, tree_predictions))
print("MAE:", mean_absolute_error(y_test, tree_predictions))
print("RMSE:", np.sqrt(mean_squared_error(y_test, tree_predictions)))


# 3. Random Forest
print("\nRandom Forest")
print("R2 Score:", r2_score(y_test, forest_predictions))
print("MAE:", mean_absolute_error(y_test, forest_predictions))
print("RMSE:", np.sqrt(mean_squared_error(y_test, forest_predictions)))


# Support Vector Regression

svm = SVR(kernel="rbf", C=100, epsilon=0.1)

svm.fit(X_train_scaled, y_train)

p4 = svm.predict(X_test_scaled)

print("\nSupport Vector Regression")
print("R2 Score:", r2_score(y_test, p4))
print("MAE:", mean_absolute_error(y_test, p4))
print("RMSE:", np.sqrt(mean_squared_error(y_test, p4)))


# 5. K-Nearest Neighbors Regression
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
p5 = knn.predict(X_test_scaled)
print("\nKNN Regression")
print("R2 Score:", r2_score(y_test, p5))
print("MAE:", mean_absolute_error(y_test, p5))
print("RMSE:", np.sqrt(mean_squared_error(y_test, p5)))

# Compare all five models

plt.bar(
    [
        "Linear Regression",
        "Decision Tree",
        "Random Forest",
        "SVM",
        "KNN"
    ],
    [
        r2_score(y_test, linear_predictions),
        r2_score(y_test, tree_predictions),
        r2_score(y_test, forest_predictions),
        r2_score(y_test, p4),
        r2_score(y_test, p5)
    ]
)

plt.ylabel("R2 Score")
plt.title("Comparison of Five Regression Models")
plt.xticks(rotation=25)
plt.tight_layout()
plt.show()




# ============================================================
# 23. VISUALIZATION 14: ACTUAL VS PREDICTED YIELD
# ============================================================

plt.figure(figsize=(9, 6))

plt.scatter(y_test, p5, alpha=0.4)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--",
    label="Perfect Prediction"
)

plt.xscale("symlog", linthresh=1)
plt.yscale("symlog", linthresh=1)

plt.title("Actual vs Predicted Crop Yield - KNN")
plt.xlabel("Actual Yield (Symmetric Log Scale)")
plt.ylabel("Predicted Yield (Symmetric Log Scale)")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 24. DISPLAY SAMPLE PREDICTIONS
# ============================================================

prediction_results = pd.DataFrame({
    "Actual Yield": y_test,
    "Predicted Yield": p5
})

print("\nSAMPLE PREDICTIONS")

print(prediction_results.head(15).round(2))


# ============================================================
# 25. FINISH
# ============================================================

print("\nCrop Yield Prediction pipeline completed successfully!")
