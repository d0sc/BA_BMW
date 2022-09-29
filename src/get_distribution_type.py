from locale import D_FMT
import scipy.stats as st
import numpy as np
from scipy.stats import norm
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(color_codes=True, style="whitegrid")
from fitter import Fitter, get_distributions, get_common_distributions

def get_best_distribution(data):
    """
    :param data: expects a 1D-Array or List: [1,34,576,312,213536,6547,75534]
    :return: best_dist: distribution type as is dist_names specified
             best_p: corresponding p-value
             params[best_dist]: corresponding parameters

    SOURCE OF CODE: https://stackoverflow.com/questions/37487830/how-to-find-probability-distribution-and-parameters-for-real-data-python-3
    """
    dist_names = ["norm", "exponweib", "weibull_max", "weibull_min", "pareto", "genextreme"]
    dist_results = []
    params = {}
    for dist_name in dist_names:
        dist = getattr(st, dist_name)
        param = dist.fit(data)

        params[dist_name] = param
        # Applying the Kolmogorov-Smirnov test
        D, p = st.kstest(data, dist_name, args=param)
        print("p value for "+dist_name+" = "+str(p))
        dist_results.append((dist_name, p))

    # select the best fitted distribution
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
    # store the name of the best fit and its p value

    print("Best fitting distribution: "+str(best_dist))
    print("Best p value: "+ str(best_p))
    print("Parameters for the best fit: "+ str(params[best_dist]))

    return best_dist, best_p, params[best_dist]


def fitter_best_distribution(df:pd.DataFrame, x):

    if False:

        # plot histogramm to get feeling for data
        sns.set_style("white")
        sns.set_context("paper", font_scale=2)
        sns.displot(data=df, x=x, kind="hist", bins=100, aspect=1.5 )

        plt.show()

    # get the columns out of dataframe and make numpy-array
    hri = df[x].values

    # fit distribution to get best one
    f = Fitter(hri, # method is sumsquare-error
               distributions=get_common_distributions())

    f.fit()
    # print("summuary: ", f.summary())

    # get best fitted-distribution and parameters
    dist_para = f.get_best(method="sumsquare_error")

    # print(f"best parameters and distribution for {x}: ", dist_para)


    return dist_para, f



if __name__ == "__main__":

    # load some somple data
    df = pd.read_csv("data/final/full_tbl_MAX.csv").iloc[:,5:20]
    # df = pd.read_csv("data/hri/HRI_f_9_f_18_further_processed.csv")

    # try different options

    
    i,j=0,0
    PLOTS_PER_ROW = 5
    
    fig, axs = plt.subplots(math.ceil(len(df.columns)/PLOTS_PER_ROW),PLOTS_PER_ROW, figsize=(18, 8))
    fig.subplots_adjust(hspace=0.4, wspace=.2)
    for col in df.columns:
        plot = sns.histplot(ax=axs[i][j], data=df, x=str(col), kde=True)
        j+=1
        if j%PLOTS_PER_ROW==0:
            i+=1
            j=0
        if j != 1:
            plot.set(ylabel=None)

            
    plt.show()
    
    

    

    # test with normal distribution
    # get_best_distribution(np.random.normal(0, 0.1, 1000))
    #print(np.random.normal(0, 0.1, 1000))
    # test with 1D-Array
    #array_1d = [1,2,3,4,5]
    #get_best_distribution(array_1d)
    