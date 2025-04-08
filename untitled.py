import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

np.random.seed(42)
x = np.linspace(10, 100, 20)  # цена от 10 до 100
y = 150 - 1.2 * x + np.random.normal(0, 10, size=len(x))  # спрос с шумом

data = pd.DataFrame({'Цена (x)': x, 'Спрос (y)': y})

model = LinearRegression()
model.fit(x.reshape(-1, 1), y)

y_pred = model.predict(x.reshape(-1, 1))

beta_0 = model.intercept_
beta_1 = model.coef_[0]
print(f"Уравнение регрессии: y = {beta_0:.2f} + {beta_1:.2f}x")

# График
plt.figure(figsize=(10, 5))
plt.scatter(x, y, label='Реальные данные')
plt.plot(x, y_pred, color='red', label='Прогноз модели')
plt.xlabel('Цена')
plt.ylabel('Спрос')
plt.title('Прогнозирование спроса на товар')
plt.legend()
plt.grid()
plt.show()