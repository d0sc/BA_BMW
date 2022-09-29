import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(color_codes=True, style="whitegrid")
import numpy as np


def read_df():

    predicted = pd.read_csv("../data/validierung_ml/eol_pred.csv").drop(columns=["Unnamed: 0"], errors='ignore')
    predicted_1 = pd.read_csv("../data/validierung_ml/eol_pred_1.csv").drop(columns=["Unnamed: 0"], errors='ignore')
    real = pd.read_csv("../data/validierung_ml/eol_compare.csv").drop(columns=["Unnamed: 0"], errors='ignore')

    df = pd.concat([predicted, real]).reset_index(drop=True).drop(columns=["Unnamed: 0"], errors='ignore')
    # print("dataframe: \n", df.head(), "\n", df.shape)

    return df, predicted, predicted_1, real


def plot_distributions(df):
    for col in df.columns[:3]:
        sns.kdeplot(df[col], shade=True, alpha=.5, linewidth=0)
        #sns.distplot(df[col])
    plt.show()



def plot_distributions_multiple_df(predicted, real):
    # fig = plt.figure(figsize=(10,6))
    # for col in predicted.columns[1:]:
    #     sns.kdeplot(predicted[col], shade=True, alpha=.5, linewidth=0)
    # for col in real.columns[1:]:
    #     sns.kdeplot(real[col], shade=True, alpha=.5, linewidth=0)

    # for col in predicted.columns[1:]:
    sns.kdeplot(data=df,
                # x=predicted[col],
                shade=True,
                alpha=.5,
                linewidth=0)
    # sns.kdeplot(data=real,
    #             # x=predicted[col],
    #             shade=True,
    #             alpha=.5,
    #             linewidth=0)

    plt.show()


if __name__=="__main__":
    df, predicted, predicted_1, real = read_df()
    plot_distributions_multiple_df(predicted, real)