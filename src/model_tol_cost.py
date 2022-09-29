from lib2to3.pgen2.token import OP
from get_distribution_type import get_best_distribution
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set_theme(color_codes=True, style="whitegrid")



def cost_from_tol_hard_approach(x0):

    try:
        
        # load in HRI columns
        # df = pd.read_csv("data/prep/max_filtered_hri_meta_eol.csv")
        df = pd.read_csv("data/prep/max_filtered_final.csv").iloc[:, 2:] 
        # print(df.columns)

        
    except Exception as e:
        print(f"Exception: {e} \n\n can NOT real in file '../data/final/having_a_look_without_nan.csv'")


    df_above = df[(df[df.columns[0]] > x0[0]) | \
                  (df[df.columns[1]] > x0[1]) | \
                  (df[df.columns[2]] > x0[2]) | \
                  (df[df.columns[3]] > x0[3]) | \
                  (df[df.columns[4]] > x0[4]) | \
                  (df[df.columns[5]] > x0[5]) | \
                  (df[df.columns[6]] > x0[6]) | \
                  (df[df.columns[7]] > x0[7]) | \
                  (df[df.columns[8]] > x0[8]) | \
                  (df[df.columns[9]] > x0[9])]
      
    

    
    def visualisierung_für_validierung(x0, df_komplett, df_toleranz_gerissen):
            
        i,j=0,0
        PLOTS_PER_ROW = 5
        PLOTS_PER_COL = 2

        fig, axs = plt.subplots(PLOTS_PER_COL, PLOTS_PER_ROW, figsize=(16, 7))
        fig.tight_layout(pad=4.4, w_pad=5.5, h_pad=5.0)
        # fig.suptitle('histogram for eac dimension of the predicted dataframe - compare to real data')
        fig.subplots_adjust(hspace=0.3, wspace=.3)
        for index, col in enumerate(df_komplett.columns):
            plot = sns.histplot(ax=axs[i][j], data=df_komplett, x=str(col), kde=True)
            axs[i][j].axvline(x0[index], 0,10, c="red")
            j+=1
            if j%PLOTS_PER_ROW==0:
                i+=1
                j=0
        # plt.savefig('Tol-cost-gerissen.svg')
        plt.show()
        fig, axs = plt.subplots(PLOTS_PER_COL, PLOTS_PER_ROW, figsize=(16, 7))
        fig.tight_layout(pad=4.4, w_pad=5.5, h_pad=5.0)
        # fig.suptitle('histogram for eac dimension of the predicted dataframe - compare to real data')
        fig.subplots_adjust(hspace=0.3, wspace=.3)
        for index, col in enumerate(df_toleranz_gerissen.columns):
            plot = sns.histplot(ax=axs[i][j], data=df_toleranz_gerissen, x=str(col), kde=True)
            axs[i][j].axvline(x0[index], 0,10, c="red")
            j+=1
            if j%PLOTS_PER_ROW==0:
                i+=1
                j=0
        # plt.savefig('Tol-cost-gerissen.svg')
        plt.show()
        
        

    # visualisierung_für_validierung(x0, df, df_above)
    
    
    # print("vorher: ", df.shape, "----", "nachher: ", df_above.shape)

    scrap = df_above.count().to_numpy()[0]
    print(f"--> Costs for Tolerances {x0} is: {scrap}")
    
    d = {'Kosten': [scrap]}
    df_scrap = pd.DataFrame(data=d)
    
    x00 = {'Toleranzvektor': [x0]}
    df_x00 = pd.DataFrame(data=x00)
    
    Opts = pd.read_csv('data/Optimierung/Kosten.csv').drop(columns=["Unnamed: 0"], errors='ignore')
    Tols = pd.read_csv('data/Optimierung/Toleranzvektor.csv').drop(columns=["Unnamed: 0"], errors='ignore')
    Opts_neu = pd.concat([Opts, df_scrap], ignore_index=True)
    Tols_neu = pd.concat([Tols, df_x00], ignore_index=True)
    
    Opts_neu.to_csv('data/Optimierung/Kosten.csv')
    Tols_neu.to_csv('data/Optimierung/Toleranzvektor.csv')

    return scrap



if __name__=="__main__":

    # x0 = np.random.randint(low=4, high=30, size=10)
    x0 =  [218.50, 236.94, 155.25, 182.02, 250.00, 187.84, 177.13, 250.00, 246.61, 250.00]
    cost_from_tol_hard_approach(x0)