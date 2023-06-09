import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')


def mean_squared_error(m, b, scores):
    total_error = 0
    for i in range(len(scores)):
        x = scores.iloc[i].GPA
        y = scores.iloc[i].SAT
        total_error += (y - (m * x + b)) ** 2
    total_error / float(len(scores))


def gradient_descent(m_now, b_now, scores, L):
    m_gradient = 0
    b_gradient = 0

    n = len(scores)

    for j in range(n):
        x = scores.iloc[j].GPA
        y = scores.iloc[j].SAT

        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))

    # print(f"mgrad {m_gradient}")
    # print(f"bgrad {b_gradient}")

    m_new = m_now - (m_gradient * L)
    b_new = b_now - (b_gradient * L)

    return m_new, b_new


m = 0
b = 0
L = 0.0001

epochs = 1000

for i in range(epochs):
    if i % 50 == 0:
        print(f"Epochs: {i}")
    m, b = gradient_descent(m, b, data, L)

print(m, b)

plt.scatter(data.GPA, data.SAT, color="black")
# plt.show()
plt.plot(list(range(2, 5)), [m * x + b for x in range(2, 5)], color="red")
plt.show()

