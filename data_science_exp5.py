import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = Path(__file__).resolve().parent
OUTPUT = BASE / "outputs"
OUTPUT.mkdir(exist_ok=True)
DATA_PATH = BASE / "urban_climate.csv"
TARGET = "urban_heat_island_intensity"

# 1. Load and inspect dataset
df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

# 2. Define independent and dependent variables
# calling_code is an identifier, so it is not used as a predictive feature.
feature_cols = [
    "latitude",
    "longitude",
    "year",
    "month",
    "temperature_celsius",
    "humidity_percent",
    "precipitation_mm",
    "wind_speed_ms",
    "country_population_reference",
    "country_area_km2"
]
X = df[feature_cols].copy()
y = df[TARGET].copy()

# 3. Basic cleaning
X = X.replace([np.inf, -np.inf], np.nan)
for col in X.columns:
    X[col] = X[col].fillna(X[col].median())
valid = y.notna()
X = X.loc[valid].reset_index(drop=True)
y = y.loc[valid].reset_index(drop=True)
print("\nCleaned feature shape:", X.shape)
print("Target:", TARGET)

# 4. Correlation analysis
train_idx, test_idx = train_test_split(
    np.arange(len(X)), test_size=0.2, random_state=42
)
X_train = X.iloc[train_idx].copy()
X_test = X.iloc[test_idx].copy()
y_train = y.iloc[train_idx].copy()
y_test = y.iloc[test_idx].copy()

train_corr = X_train.copy()
train_corr[TARGET] = y_train.values
corr_with_target = train_corr.corr(numeric_only=True)[TARGET].drop(TARGET)
corr_rank = corr_with_target.abs().sort_values(ascending=False)
print("\nAbsolute correlation with target:")
print(corr_rank)

plt.figure(figsize=(11, 8))
corr_matrix = train_corr.corr(numeric_only=True)
plt.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(OUTPUT / "01_correlation_heatmap.png", dpi=160)
plt.show()

# 5. Filter method: Mutual Information
mi_values = mutual_info_regression(X_train, y_train, random_state=42)
mi_rank = pd.Series(mi_values, index=feature_cols).sort_values(ascending=False)
print("\nMutual Information ranking:")
print(mi_rank)
mi_rank.sort_values().plot(kind="barh", figsize=(9, 6), title="Mutual Information Feature Ranking")
plt.xlabel("Mutual information")
plt.tight_layout()
plt.savefig(OUTPUT / "02_mutual_information.png", dpi=160)
plt.show()

# 6. Wrapper method: Recursive Feature Elimination (RFE)
rfe_estimator = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rfe = RFE(estimator=rfe_estimator, n_features_to_select=5, step=1)
rfe.fit(X_train, y_train)
rfe_rank = pd.DataFrame({
    "Feature": feature_cols,
    "Selected": rfe.support_,
    "RFE_Rank": rfe.ranking_
}).sort_values(["Selected", "RFE_Rank"], ascending=[False, True])
rfe_features = X_train.columns[rfe.support_].tolist()
print("\nRFE feature ranking:")
print(rfe_rank.to_string(index=False))
rfe_rank.to_csv(OUTPUT / "rfe_feature_ranking.csv", index=False)

# 7. Embedded method: Random Forest feature importance
embedded_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
embedded_model.fit(X_train, y_train)
importance = pd.Series(
    embedded_model.feature_importances_, index=feature_cols
).sort_values(ascending=False)
print("\nRandom Forest feature importance:")
print(importance)
importance.sort_values().plot(kind="barh", figsize=(9, 6), title="Random Forest Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(OUTPUT / "03_random_forest_importance.png", dpi=160)
plt.show()

# 8. Compare features selected by the three approaches
comparison = pd.DataFrame({
    "Feature": feature_cols,
    "Correlation_Top5": [f in corr_rank.head(5).index for f in feature_cols],
    "Mutual_Info_Top5": [f in mi_rank.head(5).index for f in feature_cols],
    "RFE_Selected": [f in rfe_features for f in feature_cols],
    "Random_Forest_Top5": [f in importance.head(5).index for f in feature_cols]
})
comparison["Selection_Count"] = comparison.iloc[:, 1:5].sum(axis=1)
comparison = comparison.sort_values(["Selection_Count", "Feature"], ascending=[False, True])
print("\nFeature selection comparison:")
print(comparison.to_string(index=False))
comparison.to_csv(OUTPUT / "feature_selection_comparison.csv", index=False)

# Use the five features selected by RFE for the reduced model.
selected_features = rfe_features
print("\nSelected features used for reduced model:", selected_features)

# 9-10. Train same model using all features and selected features
def evaluate_model(model_features, model_name):
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    start_train = time.perf_counter()
    model.fit(X_train[model_features], y_train)
    train_seconds = time.perf_counter() - start_train
    start_predict = time.perf_counter()
    prediction = model.predict(X_test[model_features])
    predict_seconds = time.perf_counter() - start_predict
    return {
        "Model": model_name,
        "Number_of_Features": len(model_features),
        "R2_Score": r2_score(y_test, prediction),
        "MAE": mean_absolute_error(y_test, prediction),
        "RMSE": mean_squared_error(y_test, prediction) ** 0.5,
        "Training_Time_Seconds": train_seconds,
        "Prediction_Time_Seconds": predict_seconds
    }

all_result = evaluate_model(feature_cols, "All Features")
selected_result = evaluate_model(selected_features, "RFE Selected Features")
results = pd.DataFrame([all_result, selected_result])
print("\nModel performance comparison:")
print(results.to_string(index=False))
results.to_csv(OUTPUT / "model_performance_comparison.csv", index=False)

# Model comparison graph: predictive performance and computational time
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].bar(results["Model"], results["R2_Score"])
axes[0].set_title("Model Comparison - R² Score")
axes[0].set_ylabel("R² (higher is better)")
axes[0].tick_params(axis="x", rotation=15)
axes[1].bar(results["Model"], results["Training_Time_Seconds"])
axes[1].set_title("Model Comparison - Training Time")
axes[1].set_ylabel("Training time (seconds)")
axes[1].tick_params(axis="x", rotation=15)
plt.tight_layout()
plt.savefig(OUTPUT / "04_model_comparison_graph.png", dpi=160)
plt.show()

# 11. Final interpretation
print("\nFinal interpretation:")
print("Target variable:", TARGET)
print("Original number of input features:", len(feature_cols))
print("Number of RFE-selected features:", len(selected_features))
print("RFE-selected features:", ", ".join(selected_features))
print("Compare R2 (higher is better), MAE/RMSE (lower is better), and execution times.")
print("Feature selection does not guarantee improved performance; it can reduce complexity.")

# Save cleaned data and a summary text file
cleaned = X.copy()
cleaned[TARGET] = y.values
cleaned.to_csv(OUTPUT / "urban_climate_cleaned_features.csv", index=False)
with open(OUTPUT / "experiment_summary.txt", "w", encoding="utf-8") as file:
    file.write("Data Science Experiment 5 - Feature Selection\n")
    file.write(f"Dataset shape: {df.shape}\n")
    file.write(f"Target: {TARGET}\n")
    file.write(f"Input features: {feature_cols}\n")
    file.write(f"RFE selected features: {selected_features}\n\n")
    file.write(results.to_string(index=False))
print("\nAll tables, plots and processed data are saved in the outputs folder.")
