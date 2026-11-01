# Data Science Experiment 5 – Feature Selection

## Objective
Identify relevant input features for predicting `urban_heat_island_intensity` in the supplied `urban_climate.csv` dataset. This adapts the practical's feature-selection activities to the user's urban-climate data.

## Dataset
- 11,040 rows and 20 columns.
- Target: `urban_heat_island_intensity`.
- Input features: latitude, longitude, year, month, temperature, humidity, precipitation, wind speed, country population reference and country area.
- `calling_code` is excluded because it is an identifier, not a meaningful measurement.
- City/country names and other text identifiers are not directly used in this numeric model.

## Activities implemented
1. Load and inspect the dataset.
2. Define independent variables (X) and dependent variable (y).
3. Check missing values and duplicate rows; replace missing numeric feature values with medians.
4. Analyse feature correlations and save a correlation heatmap.
5. Apply filter methods: absolute correlation ranking and Mutual Information.
6. Apply wrapper method: Recursive Feature Elimination (RFE) with a Random Forest regressor.
7. Apply embedded method: Random Forest feature importance.
8. Compare the top-five / selected features across methods.
9. Train a Random Forest regression model using all input features.
10. Train the same model using the five RFE-selected features.
11. Compare R², MAE, RMSE, training time and prediction time.

## Important methodology notes
- The target is continuous, so this is a **regression** task, not classification.
- A fixed 80:20 random train-test split is used with `random_state=42`.
- Feature selection is fitted using training data only to avoid using test labels during selection.
- RFE selects five features. The reduced model uses those same five features.
- R²: higher is better. MAE and RMSE: lower is better. Execution time depends on the computer and may vary.
- Feature selection may reduce model complexity but does not guarantee a higher R².

## Files
- `data_science_exp5.py`: complete Python script.
- `Data_Science_Experiment_5.ipynb`: Jupyter Notebook with the same workflow and explanations.
- `urban_climate.csv`: dataset used.
- `outputs/`: generated plots, comparison tables and processed data after running.

## Install packages
Run in the VS Code terminal / Command Prompt:
```bash
python -m pip install pandas numpy matplotlib scikit-learn jupyter
```

## Run the Python script
Keep the script and CSV in the same folder:
```bash
python data_science_exp5.py
```

## Run the notebook
Open `Data_Science_Experiment_5.ipynb` in Jupyter or VS Code and run all cells from top to bottom.
