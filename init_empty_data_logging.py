import pandas as pd
import numpy as np

d = {'Kosten': [173]}
df = pd.DataFrame(data=d)
print(df)
df.to_csv('data/Optimierung/Kosten.csv')


d = {'Qualität': [173]}
df = pd.DataFrame(data=d)
print(df)
df.to_csv('data/Optimierung/Qualität.csv')


d = {'Toleranzvektor':  [[218.50150215, 236.94495591, 155.25267225, 182.01819284, 250, 187.83834222, 177.13516922, 250, 246.6130606, 250]]}
df = pd.DataFrame(data=d)
print(df)
df.to_csv('data/Optimierung/Toleranzvektor.csv')