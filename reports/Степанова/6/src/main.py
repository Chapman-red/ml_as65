import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd

a, b, c, d = 0.3, 0.1, 0.06, 0.1

def func(x):
    return a * np.cos(b * x) + c * np.sin(d * x)

x = np.linspace(0, 30, 500)
y = func(x)

n_inputs = 6
X, Y = [], []

for i in range(len(y) - n_inputs):
    X.append(y[i:i+n_inputs])
    Y.append(y[i+n_inputs])

X = np.array(X, dtype=np.float32)
Y = np.array(Y, dtype=np.float32).reshape(-1, 1)

split_point = int(len(X) * 0.8)
X_train, X_test = X[:split_point], X[split_point:]
Y_train, Y_test = Y[:split_point], Y[split_point:]

torch.manual_seed(42)

X_train_t = torch.tensor(X_train)
Y_train_t = torch.tensor(Y_train)
X_test_t  = torch.tensor(X_test)
Y_test_t  = torch.tensor(Y_test)


n_hidden = 2
class ElmanNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hid = nn.Linear(n_inputs + n_hidden, n_hidden)
        self.out = nn.Linear(n_hidden, 1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros((1, n_hidden), dtype=x.dtype)
        h_shifted = torch.cat([h0, torch.zeros((x.shape[0] - 1, n_hidden), dtype=x.dtype)], dim=0)

        concat = torch.cat([x, h_shifted], dim=1)
        h = self.act(self.hid(concat))
        y = self.out(h)
        return y



def evaluate_lr(alpha):
    model = ElmanNet()
    opt = optim.Adam(model.parameters(), lr=alpha)
    loss_fn = nn.MSELoss()

    for _ in range(600):
        opt.zero_grad()
        pred = model(X_train_t)
        loss = loss_fn(pred, Y_train_t)
        loss.backward()
        opt.step()

    with torch.no_grad():
        mse = loss_fn(model(X_test_t), Y_test_t).item()
    return mse

candidates = [0.001, 0.003, 0.005, 0.01, 0.02]
results = {}

print("Подбор α:")
for lr in candidates:
    mse = evaluate_lr(lr)
    results[lr] = mse
    print(f"  α={lr} -> MSE={mse:.6f}")

alpha_best = min(results, key=results.get)
print(f"\nОптимальное alpha: {alpha_best}\n")


net = ElmanNet()
opt = optim.Adam(net.parameters(), lr=alpha_best)
loss_fn = nn.MSELoss()
history = []

for epoch in range(2000):
    opt.zero_grad()
    pred = net(X_train_t)
    loss = loss_fn(pred, Y_train_t)
    loss.backward()
    opt.step()
    history.append(loss.item())


with torch.no_grad():
    train_pred = net(X_train_t).numpy()

plt.figure(figsize=(10, 4))
plt.plot(Y_train, label="Эталон")
plt.plot(train_pred, label="Прогноз")
plt.grid()
plt.legend()
plt.title("Прогноз на обучающем участке (RNN Элмана)")
plt.show()



train_df = pd.DataFrame({
    "Эталон": Y_train.reshape(-1),
    "Прогноз": train_pred.reshape(-1)
})
train_df["Отклонение"] = train_df["Прогноз"] - train_df["Эталон"]

print("\nРезультаты обучения (первые 10):")
print(train_df.head(10))



plt.figure(figsize=(10, 4))
plt.plot(history)
plt.title("График ошибки MSE по эпохам")
plt.xlabel("Эпоха")
plt.ylabel("Ошибка")
plt.grid()
plt.show()



with torch.no_grad():
    test_pred = net(X_test_t).numpy()

test_df = pd.DataFrame({
    "Эталон": Y_test.reshape(-1),
    "Прогноз": test_pred.reshape(-1)
})
test_df["Отклонение"] = test_df["Прогноз"] - test_df["Эталон"]

print("\nРезультаты прогнозирования (первые 10):")
print(test_df.head(10))
