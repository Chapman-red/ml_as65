import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd


params = {
    "a": 0.2,
    "b": 0.4,
    "c": 0.04,
    "d": 0.4,
    "inputs": 6,
    "hidden": 2
}

epochs = 2000


def target_function(x):
    return (params["a"] * np.cos(params["b"] * x) + params["c"] * np.sin(params["d"] * x))

grid = np.linspace(0, 30, 500)
values = target_function(grid)

window = params["inputs"]
dataset_x, dataset_y = [], []

for start in range(len(values) - window):
    dataset_x.append(values[start:start + window])
    dataset_y.append(values[start + window])

dataset_x = np.asarray(dataset_x)
dataset_y = np.asarray(dataset_y)

cut = int(0.8 * len(dataset_x))

train_x = torch.tensor(dataset_x[:cut], dtype=torch.float32)
train_y = torch.tensor(dataset_y[:cut], dtype=torch.float32).reshape(-1, 1)
test_x = torch.tensor(dataset_x[cut:], dtype=torch.float32)
test_y = torch.tensor(dataset_y[cut:], dtype=torch.float32).reshape(-1, 1)

torch.manual_seed(7)


class Predictor(nn.Module):
    def __init__(self, n_in, n_hid):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.act = nn.Sigmoid()
        self.fc2 = nn.Linear(n_hid, 1)

    def forward(self, z):
        hidden = self.act(self.fc1(z))
        return self.fc2(hidden)


candidate_lrs = [0.001, 0.005, 0.01, 0.05, 0.1]
optimal_lr, best_score = None, float("inf")

print("Поиск лучшего α:")

for alpha in candidate_lrs:
    net = Predictor(params["inputs"], params["hidden"])
    loss_fn = nn.MSELoss()
    optimiz = optim.Adam(net.parameters(), lr=alpha)

    for _ in range(400):
        optimiz.zero_grad()
        y_hat = net(train_x)
        loss = loss_fn(y_hat, train_y)
        loss.backward()
        optimiz.step()

    with torch.no_grad():
        err = loss_fn(net(test_x), test_y).item()

    print(f"  α={alpha:.4f} - MSE={err:.6f}")

    if err < best_score:
        best_score = err
        optimal_lr = alpha

print(f"\nЛучшее α = {optimal_lr:.4f}  (ошибка = {best_score:.6f})\n")


model = Predictor(params["inputs"], params["hidden"])
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=optimal_lr)

loss_curve = []

for ep in range(epochs):
    optimizer.zero_grad()
    pred_train = model(train_x)
    loss_v = loss_fn(pred_train, train_y)
    loss_v.backward()
    optimizer.step()
    loss_curve.append(loss_v.item())


with torch.no_grad():
    predicted_train = model(train_x).numpy()

plt.figure(figsize=(10, 4))
plt.plot(train_y.numpy(), label="Эталон")
plt.plot(predicted_train, label="Прогноз", linestyle="--")
plt.title("Прогнозируемая функция (обучающая выборка)")
plt.grid(True)
plt.legend()
plt.show()


df_train = pd.DataFrame({
    "Эталон": train_y.numpy().flatten(),
    "Модель": predicted_train.flatten()
})
df_train["Ошибка"] = df_train["Модель"] - df_train["Эталон"]

print("\nПервые строки результата обучения:")
print(df_train.head(10))


plt.figure(figsize=(8, 4))
plt.plot(loss_curve)
plt.title("Изменение ошибки по эпохам")
plt.xlabel("Эпоха")
plt.ylabel("MSE")
plt.grid(True)
plt.show()


with torch.no_grad():
    predicted_test = model(test_x).numpy()

df_test = pd.DataFrame({
    "Эталон": test_y.numpy().flatten(),
    "Модель": predicted_test.flatten()
})
df_test["Ошибка"] = df_test["Модель"] - df_test["Эталон"]

print("\nПервые строки результата прогнозирования:")
print(df_test.head(10))
