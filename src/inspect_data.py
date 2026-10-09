import pandas as pd


data = pd.read_csv('data/houses.csv')


print(data.head())
print(data.shape)
print(data.describe())