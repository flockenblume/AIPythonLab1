import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("train.csv")

pd.set_option("display.max_columns", None)
print(df.head())
print(df.isnull().sum())

df["Age"].fillna(df["Age"].median(), inplace=True)
df["Spa"].fillna(df["Spa"].mean(), inplace=True)
df["HomePlanet"].fillna(df["HomePlanet"].mode()[0], inplace=True)

scaler = MinMaxScaler()
columns_to_scale = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

df = pd.get_dummies(df, columns=["HomePlanet"], drop_first=True)

print(df.head())

df.to_csv("processed_titanic.csv", index=False)