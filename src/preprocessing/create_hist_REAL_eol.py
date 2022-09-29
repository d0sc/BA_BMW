import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_theme(color_codes=True, style="whitegrid")
from datetime import date
today = date.today()
import json



def load_real_eol_data():
    eol = pd.read_csv("data/eol/query_eol_results.csv")
    
    #### get the axis-scaling for the eol-data
    eol_axis = eol[eol.variable == "axis"]
    eol_axis = eol_axis["curve_values"].apply(json.loads)
    first_axis = eol_axis.iloc[0]
    
    #### get the curve values and rename colums accordingly
    eol_without_axis = eol[eol.variable != "axis"]
    eol_without_axis["curve_values"] = eol_without_axis["curve_values"].apply(json.loads)   # un-jsoned array column

    ### get the curve values, with one value per column
    eol_values = eol_without_axis["curve_values"].apply(pd.Series)
    
    ### rename everything
    new_names = [f"eol_O_{x}" for x in first_axis]  # hri_O_2.98
    old_names = eol_values.columns
    eol_values.rename(columns=dict(zip(old_names, new_names)), inplace=True)


    ### concat meta-data and data, drop curve values column (array-format)
    df = pd.concat([eol_without_axis, eol_values], axis=1).drop(columns=['curve_values'], errors='ignore')
    
    #### SAFE A COPY
    df.to_csv(f"data/prep/eol_meta_values_{today}.csv")
    
     #### FILTER down the EOL data
    measurement_name = "DR-S_Spektrum_Max_Synch_Intermediate Shaft"     # (1) measurement name
    variable = "VS2_1"                                                  # (2) variable

    df = df[df['measurement_name'] == measurement_name]
    df = df[df['variable'] == variable]
    
    # Drop not needed meta-data columns
    df = df.drop(columns=["part_name",
                          "part_type",
                          "test_protocol_id",
                          "test_protocol_description",
                          "measurement_id",
                          "measurement_name",
                          "measurement_description",
                          "measurement_datetime",
                          "total_result",
                          "variable",
                          "dtype"],
                 errors='ignore')
    
    y_features = ["eol_O_5.5", "eol_O_5.75", "eol_O_19.75", "eol_O_20.0"] 
    ydf = df.loc[:, y_features]
    
    print(ydf.head())
    print(ydf.shape)
    
    ydf.to_csv(f"data/prep/eol_4_relevant_{today}.csv")


def plot_real_eol():
    
    eol = pd.read_csv("data/prep/eol_4_relevant_2022-09-14.csv").reset_index().drop(columns=["Unnamed: 0", "index"], errors='ignore')
    
    i,j=0,0
    PLOTS_PER_ROW = 2
    PLOTS_PER_COL = 2
    
    fig, axs = plt.subplots(PLOTS_PER_COL,PLOTS_PER_ROW, figsize=(18, 8))
    fig.suptitle('histogram for each dimension of the predicted dataframe - compare to real data')
    #fig.subplots_adjust(hspace=0.4, wspace=.2)
    for col in eol.columns:
        plot = sns.histplot(ax=axs[i][j], data=eol, x=str(col), kde=True)
        j+=1
        if j%PLOTS_PER_ROW==0:
            i+=1
            j=0
        if j != 1:
            plot.set(ylabel=None)
    plt.show()
    
if __name__ == '__main__':
    plot_real_eol()

