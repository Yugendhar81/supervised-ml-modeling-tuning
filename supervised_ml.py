
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

# ==========================================
# 1. CLASSIFICATION MODEL
# ==========================================
print("=== CLASSIFICATION MODEL ===")

iris = load_iris()
X_class = pd.DataFrame(iris.data, columns=iris.feature_names)
y_class = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X_class, y_class,
    test_size=0.2,
    random_state=42,
    stratify=y_class
)

classification_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

classification_params = {
    "model__C": [0.1, 1, 10]
}

classification_search = GridSearchCV(
    classification_pipeline,
    classification_params,
    cv=5,
    scoring="accuracy"
)

classification_search.fit(X_train, y_train)
class_predictions = classification_search.predict(X_test)

print("Best classification parameters:",
      classification_search.best_params_)
print("Classification accuracy:",
      round(accuracy_score(y_test, class_predictions), 4))

# ==========================================
# 2. REGRESSION MODEL
# ==========================================
print("\n=== REGRESSION MODEL ===")

diabetes = load_diabetes()
X_reg = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y_reg = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg,
    test_size=0.2,
    random_state=42
)

regression_model = RandomForestRegressor(random_state=42)

regression_params = {
    "n_estimators": [50, 100],
    "max_depth": [None, 5, 10]
}

regression_search = GridSearchCV(
    regression_model,
    regression_params,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

regression_search.fit(X_train, y_train)
reg_predictions = regression_search.predict(X_test)

mse = mean_squared_error(y_test, reg_predictions)
r2 = r2_score(y_test, reg_predictions)

print("Best regression parameters:",
      regression_search.best_params_)
print("Regression MSE:", round(mse, 2))
print("Regression R2 score:", round(r2, 4))

print("\nBoth models trained and evaluated successfully!")
