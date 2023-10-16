import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from useful_package import polynom_3, hyperbola 

x = np.linspace(1, 10, 100).reshape(-1, 1)

y_polynom = polynom_3(x).ravel()
y_hyperbola = hyperbola(x).ravel()

regressor_polynom = RandomForestRegressor()
regressor_hyperbola = RandomForestRegressor()

regressor_polynom.fit(x, y_polynom)
regressor_hyperbola.fit(x, y_hyperbola)

y_pred_polynom = regressor_polynom.predict(x)
y_pred_hyperbola = regressor_hyperbola.predict(x)

mse_polynom = mean_squared_error(y_polynom, y_pred_polynom)
mse_hyperbola = mean_squared_error(y_hyperbola, y_pred_hyperbola)

print(f"MSE for hyperbola: {mse_hyperbola}")
print(f"MSE for polynomial: {mse_polynom}")
