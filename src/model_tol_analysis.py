import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_theme(color_codes=True, style="whitegrid")
from ML_hri_eol_regression import make_prediction
from get_distribution_type import fitter_best_distribution
from sampling import sampling_flexible_distribution
from datetime import date
today = date.today()
from joblib import load
from fitter import Fitter, get_distributions, get_common_distributions



def quality_from_tol(x0):
    """
    :param x0: Current Tolerances: x0 - für anfang 11 Werte als Liste
    :return: Quality for tolerance x0 - 1 skalar Value
    """
    
    ##### get the data #####
    try:
        
        # df = pd.read_csv("data/prep/max_filtered_hri_meta_eol.csv")
        df_full = pd.read_csv("data/prep/max_filtered_final.csv")
        df = pd.read_csv("data/prep/max_filtered_final.csv").iloc[:, 2:] 
        
        #### Keep only datapoints with not surpassed HRI-tol limit        
        df_no_tol_surpassed = df[(df[df.columns[0]] < x0[0]) | \
                                 (df[df.columns[1]] < x0[1]) | \
                                 (df[df.columns[2]] < x0[2]) | \
                                 (df[df.columns[3]] < x0[3]) | \
                                 (df[df.columns[4]] < x0[4]) | \
                                 (df[df.columns[5]] < x0[5]) | \
                                 (df[df.columns[6]] < x0[6]) | \
                                 (df[df.columns[7]] < x0[7]) | \
                                 (df[df.columns[8]] < x0[8]) | \
                                 (df[df.columns[9]] < x0[9])]
        
        print(f"- APPLY FILTER for current Tolerance-vector \n\t shape vorher: {df.shape}, shape nachher: {df_no_tol_surpassed.shape}")
    
    except Exception as e:
        print(f"Exception: {e} \n\n can NOT real in file 'data/prep/max_filtered_hri_meta_eol.csv'")
        print("Tolerance-vecor: ", x0)
        print("\n before: \n")
        print("min: \n", df.min())
        print("max: \n", df.max())
        print(df.head())


    #### get the pre-trained ML model #### -> go to model and copy path from the model you want to have
    try:
        
        # MLmodel = load("model/LR/LR_model_2022-09-22_0.79_7.24_1.89.joblib") # this is a LR model
        # MLmodel = load('model/RF/RF_model_2022-09-27_0.97_5.15_1.25.joblib') #R F-model
        MLmodel = load('model/DT/DT_model_2022-09-27_0.98_7.42_1.42.joblib') #DT
        test_point = np.random.randint(low=10, high=100, size=10)
        predi = make_prediction(x=test_point.reshape(1, -1), model=MLmodel)
        print("ML-Model is working properly:", f"for point {test_point} the result {predi} was given")
    
    except Exception as e:
         print(f"Exception: {e} \n\n can NOT load the ML-Model or prediction for {test_point} was not possible")


    ##### DISTRIBUTIONS
    dist_para_for_earch_attr = {}
    x_features = df_no_tol_surpassed.columns
    
    for attr in x_features:
        try:
            # catch error when df is empty due to bad filtering
            dist_para, f = fitter_best_distribution(df_no_tol_surpassed, x=attr)
            dist_para_for_earch_attr[f"{attr}"] = dist_para
            # {'O_9.76': {'cauchy': {'loc': 6079.6075963306785, 'scale': 837.7437599981599}}, ...
            
            
            if False:
                ##### plot the Fitting distr
                # (1) plot data           
                i,j=0,0
                PLOTS_PER_ROW = 5
                PLOTS_PER_COL = 2
                
                # fig, axs = plt.subplots(PLOTS_PER_COL, PLOTS_PER_ROW, figsize=(16, 7))
                # fig.tight_layout(pad=4.4, w_pad=5.5, h_pad=5.0)
                # # fig.suptitle('histogram for eac dimension of the predicted dataframe - compare to real data')
                # fig.subplots_adjust(hspace=0.3, wspace=.3)
                # for index, col in enumerate(df_no_tol_surpassed.columns):
                # plot = sns.histplot(ax=axs[i][j], data=df_no_tol_surpassed, x=str(col), kde=True)
                # plot = sns.histplot(data=df_no_tol_surpassed, x=str(df_no_tol_surpassed.columns[2]), kde=True)
                
                f = Fitter(df_no_tol_surpassed[df_no_tol_surpassed.columns[2]], distributions=get_common_distributions())
                f.fit()
                
                plot_2 = f.plot_pdf(Nbest=3)
                
                    
                    # j+=1
                    # if j%PLOTS_PER_ROW==0:
                    #     i+=1
                    #     j=0

                plt.savefig('dist_3_fitter.svg')
                plt.show()
            
            
        except Exception as e:
            print(f"Exception: {e}")
            Quality = 0
            print(f"Quality for Tolerances {x0} is: {Quality}")
            return Quality


    ##### Now sample from distribution
    try:
        
        sample_size = 5000
        sample_list = sampling_flexible_distribution(sample_size=sample_size, **dist_para_for_earch_attr)
        
    except Exception as e:
            print(f"Exception: {e}, could not create sample list -- sample-size: {sample_size}")


    
    ##### Monte-Carlo sampling
    predictions = []    # [array([[1.49316003e+04, 2.06429833e-02, 9.00000000e+00]]),
    
    for iteration in range(sample_size): # Monte-Carlo-Loop
        
        current_sample = []
        
        for feature in sample_list: # loop over all input features
        
            try:
                # create current sample from the sample list based on iteration and feature
                object = sample_list[str(feature)][iteration]
                current_sample.append(object) # [0.5400635464325986]
            except Exception as e:
                print(f"Exception: {e}, could not get the sample from sample_list")
                
        try:
            sample = np.array(current_sample).reshape(1, -1)  # take one sample and match the shape for model input: # [[4405.94315005 5124.91560037 1524.91310893 1646.67932057]]
            predictions.append(make_prediction(x=sample, model=MLmodel))
        except Exception as e:
            print(f"Exception: {e}, could not create prediction from sampple")  


    try:
        pred_array = np.array(predictions).reshape(-1)
        pred_dataframe = pd.DataFrame(pred_array, columns = ['eol_O_20.0'])
        pred_dataframe.to_csv(f"data/results/predictions_{today}.csv")


        # try:
        #     #### get Histogram for the predictions ####
        #     def multiple_eol_features():
        #         # get a histogram for each dimension of the predicted dataframe to compare the result with the real ones from real data
        #         i,j=0,0
        #         PLOTS_PER_ROW = 2
        #         PLOTS_PER_COL = 2
                
        #         fig, axs = plt.subplots(PLOTS_PER_COL,PLOTS_PER_ROW, figsize=(18, 8))
        #         fig.suptitle('histogram for each dimension of the predicted dataframe - compare to real data')
        #         #fig.subplots_adjust(hspace=0.4, wspace=.2)
        #         for col in pred_dataframe.columns:
        #             plot = sns.histplot(ax=axs[i][j], data=pred_dataframe, x=str(col), kde=True)
        #             j+=1
        #             if j%PLOTS_PER_ROW==0:
        #                 i+=1
        #                 j=0
        #             if j != 1:
        #                 plot.set(ylabel=None)
        #         plt.show()
                
        #     sns.histplot(data=pred_dataframe, x=str(pred_dataframe.columns[0]))
        #     plt.tight_layout()
            
        #     df_eol_real = pd.read_csv("data/prep/eol_4_relevant_2022-09-14.csv").reset_index().drop(columns=["Unnamed: 0", "index"], errors='ignore')
        #     df_eol_real = df_eol_real.loc[:, ['eol_O_20.0']]
            
        #     sns.histplot(data=df_eol_real, x='eol_O_20.0', color='crimson')
        #     plt.legend(loc='upper right', labels=['gesampelte Vorhersagen', 'reelle Daten'])

        #     plt.savefig('Vorhersage_Validierung.svg')
        #     plt.show()
        #     quit()
                
        # except Exception as e:
        #     print(f"Exception: {e}, could not create histogram of predicted dataframe")  
        
        
        ######## Quality calculation ########
        eol_tolerances = [135.411]  # Prdouction-EOL-Tol for Ordnung 20.0
        df_eol_tol_gerissen = pred_dataframe[(pred_dataframe[pred_dataframe.columns[0]] > eol_tolerances[0])]
        
        print(f"- APPLY FILTER for EOL Tolerances \n\t shape vorher: {pred_dataframe.shape}, shape nachher: {df_eol_tol_gerissen.shape}")

        Ausschuss = df_eol_tol_gerissen.count().to_numpy()[0] #.sum()   # count of all parts that made it through EOL with Tolerance x0 from HRI
        
        # 5000 Samplesize
        # -> Umrechnung in ppm mit Faktor 1.000.000 / 5.000 = 200
        ppm_umrechnung = 200
        
        Quality = Ausschuss * ppm_umrechnung
        print(f"--- Quality for Tolerances {x0} is: {Quality}")
        
        d = {'Qualität': [Quality]}
        df_quali = pd.DataFrame(data=d)
        Opts = pd.read_csv('data/Optimierung/Qualität.csv').drop(columns=["Unnamed: 0"], errors='ignore')
        Opts_neu = pd.concat([Opts, df_quali], ignore_index=True)
        Opts_neu.to_csv('data/Optimierung/Qualität.csv')
        # print(Opts_neu)
        
        
        Q_min = 2000 # define minimum value for Quality
        Constraint = Quality - Q_min        #FIXME: Q_min noch nicht definiert -> done
        print(f"Return Contraint - Tol-Analy - {Constraint}")
        
        return Constraint
    
    
    except Exception as e:
        print(f"Exception: {e}, could not create and safe predictions as Dataframe") 
        print(f"Exception: {e}, could filter the predicted eol dataframe") 
        print(f"Exception: {e}, could not calculate the Quality")


if __name__=="__main__":

    # x0 = np.random.randint(low=50, high=150, size=10)
    # x0 = [3, 5, 12, 13, 1, 2, 5, 34, 30, 21]
    # x0 = [7, 8, 10, 10, 5, 5, 14, 60, 20, 20]
    # x0 =  [214.37733067, 88.23696861, 106.17721566, 187.23929487, 51.78622418, 51.93238754, 125.02008592, 89.04855188, 40.4965299, 207.5162935 ]
    x0 =  [218.50, 236.94, 155.25, 182.02, 250.00, 187.84, 177.13, 250.00, 246.61, 250.00]
    
    quality_from_tol(x0)