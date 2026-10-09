
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def train_and_evaluate(data_path):
    # 1. Load the dataset
    data = pd.read_csv(data_path)

    # 2. Separate features and target
    X = data[["house_size_sqft"]]
    y = data["price_lakh"]

    # 3. Split the data
    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # 4. Train the model
    model = LinearRegression()
    model.fit(x_train, y_train)

    # 5. Predict prices for test houses
    y_pred = model.predict(x_test)

    # 6. Calculate evaluation metrics
    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }

    return model, metrics


def main():
    model, metrics = train_and_evaluate("data/houses.csv")

    print(f"Slope: {model.coef_[0]}")
    print(f"Intercept: {model.intercept_}")

    print(f"MAE: {metrics['mae']}")
    print(f"MSE: {metrics['mse']}")
    print(f"R2: {metrics['r2']}")


if __name__ == "__main__":
    main()
