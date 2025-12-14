import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(
    "auto-mpg.csv",
    na_values="?"
)

print("Размер датасета:", data.shape)
print(data.head(), "\n")



data["horsepower"] = data["horsepower"].astype(float)

hp_avg = data["horsepower"].mean()
data["horsepower"].fillna(hp_avg, inplace=True)

print(f"Заполненные пропуски horsepower средним: {hp_avg:.2f}\n")



plt.figure(figsize=(8, 5))
plt.scatter(
    data["weight"],
    data["mpg"],
    alpha=0.7
)
plt.title("mpg в зависимости от веса автомобиля")
plt.xlabel("Weight")
plt.ylabel("MPG")
plt.grid(True)
plt.show()



origin_labels = {
    1: 0,
    2: 1,
    3: 2
}

data["origin"] = data["origin"].replace(origin_labels)
print("Уникальные значения origin:", data["origin"].unique(), "\n")



collection_year = 1983
data["age"] = collection_year - (data["model year"] + 1900)

print(data[["model year", "age"]].head(), "\n")



cyl_counts = data["cylinders"].value_counts().sort_index()

plt.figure(figsize=(6, 4))
plt.bar(
    cyl_counts.index.astype(str),
    cyl_counts.values
)
plt.title("Распределение автомобилей по количеству цилиндров")
plt.xlabel("Cylinders")
plt.ylabel("Количество")
plt.show()
