from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel(r'umbrellas_sold_1.xlsx')

x = df['rainfall_mm']
y = df['umbrellas_sold']

plt.figure(figsize=(4, 3))
plt.scatter(x, y, color='blue', label='Actual data')

x = np.array(x).reshape(-1, 1)

model = LinearRegression()
model.fit(x, y)

accuracy = model.score(x, y)
print(f'accuracy {accuracy:.2f}')

# Predict y values for the existing x values to plot the regression line
y_pred = model.predict(x)
# Plot the regression line
plt.plot(x, y_pred, color='red', label='Prediction line')

plt.xlabel('Rainfall (mm)')
plt.ylabel('Umbrellas Sold')
plt.legend()
plt.show()

# Print the regression equation
ic = '{:.2f}'.format(model.intercept_)
ce = '{:.2f}'.format(model.coef_[0])
print(f'สมการทำนายผลคือ: y = {ic} + ({ce})x')

# Predict umbrellas sold for specific rainfall values
x_predict = [[90], [100], [120]]
y_predicted = model.predict(x_predict)

print()

# Display the prediction results
for (i, p) in enumerate(x_predict):
    sale = '{:.0f}'.format(y_predicted[i])
    print(f'ปริมาณฝนที่ตก {p[0]} มิลลิเมตร จะขายร่มได้ {sale} อัน')
