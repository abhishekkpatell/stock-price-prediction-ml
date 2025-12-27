# Stock Price Prediction ML

This repository contains a Python machine learning project to predict stock prices using historical datasets. It was completed as part of the Python Developer Intern Assignment (2026).

The goal of this project is to model how changes in previous-day data influence stock price movements.

---

## Approach and Assumptions

### Problem Approach

1. Load the independent and dependent datasets.
2. Merge data by date and align chronologically.
3. Engineer features that capture previous-day change and returns.
4. Train models on engineered features to predict:
   - stock price levels
   - stock price returns and movement direction
5. Evaluate performance using regression metrics and directional accuracy.
6. Visualize results for interpretation.

### Key Assumptions

- Stock price movement is primarily driven by changes in the previous day’s data.
- Other external factors (e.g., market sentiment, news, macro indicators) are ignored.
- Time order is preserved to ensure no data leakage.
- Return-based features help stabilize price movements for modeling.

---

## Data Preprocessing Steps

1. **Load and Align Datasets**
   - Both datasets are read based on a common date field.
   - Chronological sorting ensures time order is preserved.

2. **Handle Missing Values**
   - Forward-fill and backward-fill are applied.
   - Rows with remaining missing values are removed.

3. **Feature Engineering**
   - **Lag Features:** Previous-day values for key variables.
   - **Absolute Change:** Day-over-day difference.
   - **Percentage Return:** Percentage change compared to the previous day.
   - **Direction Label:** 1 for upward movement, 0 for downward movement.

4. **Scaling**
   - StandardScaler is applied for Ridge Regression.
   - Scale is not used for tree-based models.

5. **Train-Test Split**
   - Data is split into training (first 80%) and testing (last 20%) without shuffling to maintain time dependency.

---

## Model Selection and Evaluation

### Models Used

- **Ridge Regression**
  - A linear model that provides interpretability and serves as a baseline.

- **Histogram-Based Gradient Boosting (HGB) Regressor**
  - A tree-based ensemble method that captures nonlinear patterns.

Both models are trained on:
- Return-level targets
- Price-level targets

### Evaluation Metrics

- **Mean Squared Error (MSE)** — average squared difference between predictions and actual values.
- **R-squared (R²)** — proportion of variance explained by the model.
- **Directional Accuracy** — the percentage of correct up/down movement predictions.

### Model Results

#### Return-Level Results
- Ridge | Return MSE: `0.00014448044367255928`
- Ridge | Return R²: `-0.41641941228660717`
- HGB   | Return MSE: `0.00010212384609577043`
- HGB   | Return R² : `-0.0011749299112377987`

#### Price-Level Results
- Ridge | Price MSE: `3059.8211992646197` | Price R²: `0.9920891551194578`
- HGB   | Price MSE: `2317.6158207714684` | Price R²: `0.9940080488182709`

#### Directional Accuracy
- Ridge: `0.5351925630810093`
- HGB  : `0.5272244355909694`

---

## Key Insights and Conclusions

1. **Strong Price-Level Prediction**
   - Both models achieve R² values above 0.99 for price prediction, indicating excellent capacity to predict stock price levels.

2. **Gradient Boosting Performs Slightly Better**
   - HGB achieves lower MSE and slightly higher R² for price-level prediction than Ridge.

3. **Return Prediction is Difficult**
   - Return-level R² values are negative or near zero, indicating returns are hard to model using only the previous day’s data.

4. **Directional Accuracy is Realistic**
   - Both models achieve around 52–54% directional accuracy. This is slightly better than random guessing and indicates a modest predictive signal.

5. **Feature Engineering Validates Core Assumption**
   - Deriving features based on previous-day changes contributes useful predictive information, especially for price-level regression.

---

## Visualizations

The notebook includes plots for:
- Actual vs Predicted price curves
- Return prediction error visualization
- Direction movement comparison

These facilitate deeper insight into model performance and limitations.

---

## Technology Stack

- Python
- NumPy & Pandas
- Scikit-Learn
- Matplotlib / Seaborn
- Jupyter Notebook

---

## How to Run

1. Install project dependencies:

pip install -r requirements.txt

2. Open the Jupyter notebook:

jupyter notebook Assignment.ipynb

3. Run all cells to:
- preprocess the data
- engineer features
- train the models
- evaluate performance
- generate visualizations

