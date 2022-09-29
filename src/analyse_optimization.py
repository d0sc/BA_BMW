import xdrlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
sns.set_theme(color_codes=True, style="whitegrid")


cost = pd.read_csv('data/Optimierung/Kosten.csv').drop(columns=["Unnamed: 0"], errors='ignore')
quali = pd.read_csv('data/Optimierung/Qualität.csv').drop(columns=["Unnamed: 0"], errors='ignore')
quali = quali.rename(columns={'Qualität': 'Ausschuss in ppm'})

df = pd.concat([cost, quali], axis=1)
df.reset_index()
x = df.index
print(df.head())

# plot = sns.lineplot(data=df).twiny()



ax = sns.lineplot(data=df, x=x, y=df.columns[0], color="r", legend=False)
# ax2 = ax.twinx()
# ax.set(ylim=(0, 500))
# ax2 = sns.lineplot(data=df, x=x, y=df.columns[1], color="b", ax=ax2, legend=False)
# # ax.figure.legend()
# plt.legend(loc='upper right', labels=['Kosten', 'Ausschuss in ppm'])


# ax = df.plot(x="date", y="column1", legend=False)
# ax2 = ax.twinx()
# df.plot(x="date", y="column2", ax=ax2, legend=False, color="r")
# ax.figure.legend()
# plt.show()


# plt.savefig('Optimierugsergebnisse.svg')
plt.show()