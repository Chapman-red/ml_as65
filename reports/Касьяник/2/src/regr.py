import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np

cars = pd.read_csv("CarPrice_Assignment.csv")

features = [
    "horsepower",
    "enginesize",
    "curbweight",
    "citympg",
    "highwaympg"
]

X = cars[features]
y = cars["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

reg = LinearRegression()
reg.fit(X_train, y_train)

predictions = reg.predict(X_test)

r2_value = r2_score(y_test, predictions)
mae_value = mean_absolute_error(y_test, predictions)

print(f"R² = {r2_value:.4f}")
print(f"MAE = {mae_value:.2f}")

hp = cars["horsepower"].values.reshape(-1, 1)
price = cars["price"].values

simple_model = LinearRegression()
simple_model.fit(hp, price)

x_line = np.linspace(hp.min(), hp.max(), 100).reshape(-1, 1)
y_line = simple_model.predict(x_line)

plt.figure(figsize=(7, 5))
plt.scatter(hp, price, alpha=0.6)
plt.plot(x_line, y_line)
plt.xlabel("Horsepower")
plt.ylabel("Price")
plt.title("Цена автомобиля в зависимости от мощности")
plt.show()
