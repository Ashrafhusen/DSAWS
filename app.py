import os
import sys
import pandas as pd
import numpy as np
from urllib.parse import urlparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import ElasticNet
import mlflow
import mlflow.sklearn
import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metric(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":

    # Load dataset
    csv_url = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception("Unable to download training & test CSV")

    train, test = train_test_split(data)

    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    # Set tracking URI
    os.environ["MLFLOW_TRACKING_URI"] = "http://ec2-13-217-164-166.compute-1.amazonaws.com:5000/"
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    # Define parameter grid
    param_grid = {
        "alpha": [0.01, 0.1, 0.5, 1.0, 5.0],
        "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]
    }

    # Run Grid Search
    grid_search = GridSearchCV(
        estimator=ElasticNet(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )

    grid_search.fit(train_x, train_y.values.ravel())
    best_model = grid_search.best_estimator_

    predicted_qualities = best_model.predict(test_x)
    (rmse, mae, r2) = eval_metric(test_y, predicted_qualities)

    # MLflow Logging
    with mlflow.start_run():
        mlflow.log_param("best_alpha", grid_search.best_params_["alpha"])
        mlflow.log_param("best_l1_ratio", grid_search.best_params_["l1_ratio"])

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        print("Tuned ElasticNet model:")
        print("  Best alpha:", grid_search.best_params_["alpha"])
        print("  Best l1_ratio:", grid_search.best_params_["l1_ratio"])
        print("  RMSE:", rmse)
        print("  MAE:", mae)
        print("  R2:", r2)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(best_model, "model", registered_model_name="ElasticNetTuned")
        else:
            mlflow.sklearn.log_model(best_model, "model")
