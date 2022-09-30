from lib2to3.pgen2.pgen import DFAState
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## TODO: do the same for min(), mean()
df = pd.read_csv("data/final/full_tbl_MAX.csv")


def get_filtered_dataframe(df):
    df = df.loc[:, ["eol_O_20.0", 
                    "hri_O_9.9", "hri_O_13.31", "hri_O_19.91","hri_O_20.02", "hri_O_21.3",
                    "hri_O_47.18", "hri_O_78.27", "hri_O_78.38", "hri_O_86.15", "hri_O_86.26"]]
    # colum_names = df.columns
    # df = df.rename(columns={colum_names[0]: "e_"+colum_names[0][6:],
    #                         colum_names[1]: "h_"+colum_names[1][6:],
    #                         colum_names[2]: "h_"+colum_names[2][6:],
    #                         colum_names[3]: "h_"+colum_names[3][6:],
    #                         colum_names[4]: "h_"+colum_names[4][6:],
    #                         colum_names[5]: "h_"+colum_names[5][6:],
    #                         colum_names[6]: "h_"+colum_names[6][6:],
    #                         colum_names[7]: "h_"+colum_names[7][6:],
    #                         colum_names[8]: "h_"+colum_names[8][6:],
    #                         colum_names[9]: "h_"+colum_names[9][6:],
    #                         colum_names[10]: "h_"+colum_names[10][6:]})
    
    print(df.head())
    # dataplot = sns.heatmap(df.corr().abs(), cmap="YlGnBu", annot=False)
    # plt.show()
    df.to_csv("data/prep/max_filtered_final.csv") 
    return df


def vis_heat_komplett_alle_Ordnungen(df:pd.DataFrame, printvalue:bool=True):
    
    ### rename the value columns according to respective Ordnung
    old_names = [x for x in df.columns[1:]]
    new_names = [f"{x[6:]}" for x in df.columns[1:]]
    df.rename(columns=dict(zip(old_names, new_names)), inplace=True)
    df = df.reset_index()
    
    sns.heatmap(df.corr().abs().iloc[3:850, 856:],
                           cmap="YlGnBu",
                           annot=False,
                        #    yticklabels=False,
                        #    xticklabels=False
                           )
    plt.savefig('Full_korr_3000.png', dpi=3000)
    plt.show()
    
    
def visualize_linear_regression(df, x_features, y_features):

    count_plots_horizontal = len(x_features)
    count_plots_vertikal = len(y_features)

    fig, axes = plt.subplots(count_plots_vertikal, count_plots_horizontal, figsize=(18, 8))
    fig.suptitle('Correlation-plots for all HRI-Ordnungen to all EOL-Ordnungen')
    fig.subplots_adjust(hspace=.5, wspace=.001)

    for row in range(count_plots_horizontal):
        for col in range(count_plots_vertikal):

            plot = sns.regplot(ax=axes[col, row],
                        x=df.loc[:, x_features[row]],
                        y=df.loc[:, y_features[col]],
                        data=df)
            plot.set(xticklabels=[], yticklabels=[])
            
            if col != 4:
                plot.set(xlabel=None)
            if row != 0:
                plot.set(ylabel=None)

    plt.show()
    

def create_hri_eol_plot(df_in:pd.DataFrame, hri_Ordnung:str, eol_Ordnung:str):
    
    eol_features = ["eol_O_7.0", 
                    "eol_O_10.0", 
                    "eol_O_13.75", 
                    "eol_O_20.0", 
                    "eol_O_20.75"]

    hri_features = ["hri_O_19.38", 
                    "hri_O_19.7",
                    "hri_O_19.81", 
                    "hri_O_20.13", 
                    "hri_O_20.23"]
    
    visualize_linear_regression(df=df_in, x_features=hri_features, y_features=eol_features)
    
    


    

def Korrelationsanalyse_nach_web(df_start):
    df_eol = df_start.loc[:, ["eol_O_6.75", 
                            "eol_O_7.0", 
                            "eol_O_10.0", 
                            "eol_O_20.0",
                            "eol_O_13.75", 
                            "eol_O_14.0"]]
    
    # df_eol = df_start.iloc[:, 856:]
    df_hri = df_start.iloc[:, 3:852]
    eol_len = len(df_eol.columns)
    hri_len = len(df_hri.columns)
    
    df = pd.concat([df_eol, df_hri], axis=1, join="inner")
    
    ### rename the value columns according to respective Ordnung
    old_names = [x for x in df.columns]
    new_names = [f"{x[6:]}" for x in df.columns]
    df.rename(columns=dict(zip(old_names, new_names)), inplace=True)
    
    
    corr_df = df.corr().abs().iloc[:eol_len, eol_len:hri_len+eol_len]
    
    threshold = 0.7
    for col in corr_df.columns:
        if float(corr_df[col].max()) <= float(threshold):
            corr_df = corr_df.drop([col], axis=1)
    
    dataplot = sns.heatmap(corr_df,
                        cmap="YlGnBu",
                        annot=False)
    plt.savefig('Koo_small.svg')
    plt.show()
    quit()


    def get_redundant_pairs(df):
        '''Get diagonal and lower triangular pairs of correlation matrix'''
        pairs_to_drop = set()
        cols = df.columns
        for i in range(0, df.shape[1]):
            for j in range(0, i+1):
                pairs_to_drop.add((cols[i], cols[j]))
        return pairs_to_drop

    def get_top_abs_correlations(corr_df, n):
        au_corr = corr_df.abs().unstack()
        labels_to_drop = get_redundant_pairs(df)
        au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
        au_corr.to_csv("data/results/corr_results.csv")
        return au_corr[:]

    print("Top Absolute Correlations")
    print(get_top_abs_correlations(corr_df, 3))



# Korrelationsanalyse_nach_web(df)
# create_hri_eol_plot(df, "hri_O_19.28","eol_O_7.0")
vis_heat_komplett_alle_Ordnungen(df)
# get_filtered_dataframe(df)