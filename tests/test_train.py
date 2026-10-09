
import math

from sklearn.linear_model import LinearRegression

from train import train_and_evaluate



def test_training_returns_model_and_metrics():
    model, metrics = train_and_evaluate("data/houses.csv")

    # Check that we received a trained Linear Regression model.
    assert isinstance(model, LinearRegression)
    assert hasattr(model, "coef_")

    # Check that all expected metrics are returned.
    assert set(metrics.keys()) == {"mae", "mse", "r2"}

    # Check that the metric values are valid numbers.
    for value in metrics.values():
        assert isinstance(value, float)
        assert math.isfinite(value)
