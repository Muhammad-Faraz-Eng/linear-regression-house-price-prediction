# House Price Prediction — Linear Regression

A beginner-friendly Machine Learning project that predicts house prices from house size using Python and scikit-learn.

This project is part of my journey to learn Machine Learning through practical projects, with a focus on understanding how models work, why they are used, how to evaluate predictions, and how to test and automate ML code.

## Project Objectives

- Understand the fundamentals of supervised Machine Learning.
- Learn the relationship between features and targets.
- Train a Linear Regression model.
- Predict prices for unseen examples.
- Evaluate predictions using MAE, MSE, and R².
- Write automated tests using pytest.
- Add Continuous Integration (CI) using GitHub Actions.

## Technology Stack

- Python
- pandas
- scikit-learn
- uv
- pytest
- GitHub Actions (planned)

## Dataset

The project uses a small, illustrative CSV dataset containing:

| Column | Description |
|---|---|
| `house_size_sqft` | House size in square feet; the input feature |
| `price_lakh` | House price in lakhs; the target variable |

**Note:** The dataset contains practice values and is not intended for real-world property valuation.

## How the Model Works

The project follows these steps:

1. Load the dataset using pandas.
2. Separate the input feature (`X`) from the target (`y`).
3. Split the data into training and testing sets using an 80/20 split.
4. Train a Linear Regression model on the training data.
5. Predict house prices for the testing data.
6. Evaluate the predictions using MAE, MSE, and R².

Linear Regression learns a relationship represented by:

`price = slope × house_size + intercept`

The learned parameters are used to generate predictions for new house sizes.

## Project Structure

```text
linear-regression-house-price/
├── data/
│   └── houses.csv
├── src/
│   └── train.py
├── tests/
│   └── test_train.py
├── README.md
├── pyproject.toml
└── uv.lock
```

## Getting Started

### Prerequisites

- Python
- uv

### Install dependencies

Clone the repository and move into its directory:

```bash
git clone <YOUR_REPOSITORY_URL>
cd linear-regression-house-price-prediction
```

Install the project dependencies:

```bash
uv sync
```

### Run the model

```bash
uv run python src/train.py
```

The script prints the learned slope and intercept, along with the evaluation metrics.

### Run automated tests

```bash
uv run pytest -v
```

The tests verify that the training function returns a fitted Linear Regression model and valid evaluation metrics.

## Current Evaluation Results

Results from the current practice dataset and fixed train/test split:

| Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | 5.9138 lakh |
| Mean Squared Error (MSE) | 35.0354 lakh² |
| R² score | 0.9935 |

These results are based on only two test examples. They demonstrate the evaluation workflow, not the model's expected performance on real housing data. The R² score should not be interpreted as percentage accuracy.

## Testing and CI

The project includes a pytest test for the training function.

GitHub Actions CI will be configured to run the automated tests when code is pushed to GitHub or a pull request is opened.

## What I Learned

- Features and targets in supervised learning
- Training versus testing data
- Linear Regression and its learned parameters
- Generating predictions using `predict()`
- Evaluating models using MAE, MSE, and R²
- Refactoring code into reusable functions
- Writing automated tests with pytest

## Future Improvements

- Automate testing with GitHub Actions.
- Improve test coverage.
- Experiment with larger datasets.
- Compare Linear Regression with other Machine Learning models.

## Disclaimer

This is an educational project. The dataset is illustrative and the model is not suitable for actual property pricing decisions.