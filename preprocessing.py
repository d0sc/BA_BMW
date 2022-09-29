#########################################################
### Get and prepare all the data for main.py to start ###
#### Rückgabewert: Dataframe with ALL relevant data #####
#########################################################


from os import walk
import boto3
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sagemaker import get_execution_role
import pandas as pd
import numpy as np
import json
import random
from datetime import date
today = date.today()
sns.set_theme(color_codes=True, style="whitegrid")


def get_noticable_ids_from_excel():

    """
    :return: the DMC-code for the troublesome heats
    """

    relevant_ids = pd.read_excel('data/hri/hri_ord_19_8_max_002.xlsx',
                                 engine='openpyxl')#.iloc[:, :-3]

    relevant_ids.set_axis(["part_id",
                           "engine_id",
                           "Amplitude_19_8_gr_100",
                           "spindle",
                           "Date"],
                           axis=1,
                           inplace=True)

    return relevant_ids


def get_hri_fft_data_from_s3():

    """
    :return: hri-fft-data as dataframe
    """

    # connect to s3 bucket
    bucket = 'gear-honing-heat'
    prefix = 'Maschinendaten/500_Honen/0010594382/HriFFTLog/'
    role = get_execution_role() # get standart permissions for reading files in s3
    s3client = boto3.client('s3')
    paginator = s3client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    DataFrames = []  # valid, because not much data
    k = 0

    # relevant_ids = pd.read_excel('data/hri/hri_ord_19_8_max_002.xlsx', engine='openpyxl').iloc[:,:-3]
    # part_ids = list(relevant_ids.part_id)
    # print("relevant part id´s: ", part_ids[:])

    for page in pages:
        for obj in page['Contents']:

            if obj['Key'] == 'Maschinendaten/500_Honen/0010594382/HriFFTLog/':
                continue

            file = obj['Key']

            if k < 630:
                k += 1
                continue
                # not data in this timeframe relevamt

            try:
                df = pd.read_csv(f's3://gear-honing-heat/{file}',
                                 sep=",",
                                 skiprows=list(range(0, 31))
                                 )

                df_meta = pd.read_csv(f's3://gear-honing-heat/{file}',
                                      sep=",",
                                      header=None,
                                      nrows=6
                                      )

                # add metadata to data-dataframe
                meta_list = df_meta.iloc[:, 0].tolist()
                df.insert(0, "processtep", str(int(file[-9])))
                #df.insert(0, "file_type", meta_list[0])
                #df.insert(0, "id", meta_list[1])
                #df.insert(0, "part_number", meta_list[2])
                df.insert(0, "spindle", meta_list[3])
                df.insert(0, "increment", meta_list[5])

                df = df.iloc[1:, :-1]  # Drop first row and last column of a dataframe
                df["DMC"] = df.DMC.str[:-2]

                # print("before all: \n", df.head())

                # timestamp -> datetime format
                df.insert(0, "datetime", pd.to_datetime(df["TimeStamp"], unit='ms'))

                # print("with correct date: \n", df.head())

                # df = df.drop(["TimeStamp"], axis=1)
                lower_date = "2022-02-08"
                upper_date = "2022-02-16"
                df = df[(df["datetime"] >= lower_date) & (df["datetime"] <= upper_date)]

                # print("after date filter: \n", df.head())

                #df = df.loc[df.DMC.isin(part_ids)]

                # drop unneded meta-data columns
                df = df.drop(columns=["Unnamed: 0",
                                      #"id",
                                      #"file_type",
                                      "datetime",
                                      #"part_number",
                                      "TimeStamp",
                                      "Alarmlevel",
                                      "TeilNr"],
                             errors='ignore')

                # df = df.iloc[:, :50]  # cut away columns for reduced memory


                df = df.groupby(['DMC', 'spindle', 'processtep', 'Drehzahl', 'increment']).max().reset_index() # Aggregation
                df = df[(df['spindle'] == 'C1-Spindle') | (df['spindle'] == 'C2-Spindle')]  # filter for correct spindle


                DataFrames.append(df)
                del df_meta, df

            except Exception as e:
                print(f"cant read file - {file}")
                print(f"Exception - {e}")

            k += 1

            if k % 5 == 0:
                print("Iteration: ", k)

            if k % 30 == 0:
                temp = pd.concat(DataFrames)
                temp.to_csv(f"data/hri/hri_iteration_27_07_2022_MAX/iter_{k}.csv")
                del DataFrames
                DataFrames = []


def stack_hri():
    """
    NO Processign here
    :return: stacked dataframe of all dataframes inside path
    """

    filenames = next(walk("data/hri/hri_iteration_27_07_2022_MAX"), (None, None, []))[2]
    DataFrames = []

    for iter, filename in enumerate(filenames):
        df = pd.read_csv(f"data/hri/hri_iteration_27_07_2022_MAX/{filename}")
        df_shape = df.shape

        if df_shape[0] < 100:
            continue

        DataFrames.append(df)
        del df

    df = pd.concat(DataFrames)  # (67975, 8)
    del DataFrames

    df.to_csv(f"data/hri/all_stacked_MAX.csv")  # save for savety
    print("stacked Dataframe: " + "\n", df.head())

    return df



def next_operation(df):
    # accept df as input from stack_hri()-method !
    # df = pd.read_csv(f"data/hri/all_stacked_MAX.csv")

    df = df[df["Drehzahl"] == -5502]    # only get only rpm, due to change and resulting difference in Ordnungsspektrum

    # drop Unnamed column
    df = df.drop(columns=["Unnamed: 0",
                          "Unnamed: 0.1",
                          "Unnamed: 0.2",
                          "Unnamed: 0.3",
                          "Unnamed: 0.4",
                          "Unnamed: 0.5",
                          "Unnamed: 0.6"],
                 errors='ignore')

    # here are only C1 and C2 spindle values
    df = df.groupby(['DMC',
                     'Drehzahl',
                     'increment']).max().reset_index().drop(columns=["processtep", "spindle"],
                                                             errors='ignore')

    print("averaged over all processteps: " + "\n", df.head())

    print("unique rpm: ", df.Drehzahl.unique())

    df = df.melt(['DMC',
                  'Drehzahl',
                  'increment'],
                 var_name='Wert',
                 value_name='Value')

    df["Wert"] = pd.to_numeric(df["Wert"].str[4:],
                               downcast="integer")  # get the integer behind "Wert"

    df["Ordnung"] = (df["Wert"] * df["increment"]) * 60 / abs(df["Drehzahl"])  # FFT * 60 / n   60 because min->sec

    df = df.drop(columns=["Unnamed: 0",
                          "Wert",
                          "Drehzahl",
                          "increment"],
                 errors='ignore')

    df = df.iloc[:-1, :]

    df = df.groupby(['DMC', 'Ordnung']).max().reset_index()
    df = df[df["DMC"] != "FALSE     "]  # filter False values
    df = df[df["DMC"] != "NoSeri"]      # Filter NoSeri values

    df = df.pivot(index='DMC',
                  columns='Ordnung',
                  values='Value')

    print("Ordnungstransformation done: " + "\n", df.head())

    ### rename the value columns according to respective Ordnung
    old_names = [x for x in df.columns[1:]]
    new_names = [f"hri_O_{round(float(x), 2)}" for x in df.columns[1:]]
    df.rename(columns=dict(zip(old_names, new_names)), inplace=True)
    df = df.reset_index()

    print("renamed: " + "\n", df.head())

    df.to_csv("data/hri/renamed_MAX.csv")   # save

    return df





def full_join_with_excel_data():

    df_hri = pd.read_csv("data/hri/hri_6200_fp.csv")    # NO dublicated DMC codes !!!! prooved

    df_noti_ids = pd.read_excel('data/hri/hri_ord_19_8_max_002.xlsx', engine='openpyxl')  # read in Bastians_excel
    df_noti_ids.rename(columns={"part_id": "DMC"}, inplace=True)

    df = pd.merge(how='outer',
                      left=df_hri,
                      right=df_noti_ids,
                      left_on='DMC',
                      right_on='DMC')

    df = df.drop(columns=["Unnamed: 0",
                          "spindle",
                          "part_datetime"],
                 errors='ignore')

    print(df.columns)
    print(df.head())

    df.to_csv(f"data/hri/merged_hri_6200_excel_bastian.csv")





def get_eol_process_and_join(hri):
    """
    :input:  preprocessed HRI data with: DMC / Ordnungen...
    :return: eol-data from the csv, which is the result from the huge query in athena
    """

    ### load in the data
    eol = pd.read_csv("data/eol/query_eol_results.csv")                 # EOL-data from query as csv
    mapper = pd.read_excel("data/hri/dmc_heat_zuordnung.xlsx")          # mapper from bastian, gives rel between DMC & Heat
    # hri = pd.read_csv("data/hri/renamed_MAX.csv").drop(columns=["Unnamed: 0"], errors='ignore')


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

    # Join HRI-data and mapper
    merge_hri_mapper = pd.merge(how='left',
                               left=hri,
                               right=mapper,
                               left_on='DMC',
                               right_on='zw_dmc')

    # Join mapper and EOL-data
    result = pd.merge(how='left',
                     left=merge_hri_mapper,
                     right=df,
                     left_on='engine_id',
                     right_on='part_id')


    ### check how much data we realy have
    result.replace("", float("NaN"), inplace=True) # drop rows, where values are missing
    result.dropna(subset=["part_id"], inplace=True)
    result.to_csv("data/prep/joined_hri_mapper_eol.csv")   # make a safety copy
    result.to_csv("data/final/full_tbl_MAX.csv")   # make a safety copy


    return result


def filter_to_needed(df):
    #df = pd.read_csv(f"data/final/full_tbl_MAX.csv")
    print("#########################################")
    print("#########################################")
    print("#########################################")
    print(df.head())
    print("#########################################")
    print("#########################################")
    print("#########################################")

    df = df.loc[:, ["DMC",
                    "hri_O_19.28", "hri_O_19.38", "hri_O_19.49","hri_O_19.6", "hri_O_19.7",
                    "hri_O_19.81",
                    "hri_O_19.91", "hri_O_20.02", "hri_O_20.13", "hri_O_20.23", "hri_O_20.34",
                    "eol_O_5.5", "eol_O_5.75", "eol_O_19.75", "eol_O_20.0"]]

    print(df.head())

    df.to_csv("data/final/filtered_to_rel_Ordn_MAX.csv")

    return df



def validierung_hri_data():


    filenames = next(walk(f"data/log_data"), (None, None, []))[2]

    DataFrames = []

    for iter, filename in enumerate(filenames):

        df = pd.read_csv(f"data/log_data/{filename}", sep=";")

        df_shape = df.shape

        if df_shape[0] < 100:
            continue

        DataFrames.append(df)
        del df


    df = pd.concat(DataFrames)  # (67975, 8)
    del DataFrames

    rel_col = ["C1-Spindle_Ord19.8_bw1_h1_n-5502",
               "C2-Spindle_Ord19.8_bw1_h2_n-5502"]

    meta_col = ["DMC"]

    df = df.groupby(['DMC']).mean().reset_index()

    df = df.loc[:, meta_col + rel_col]

    df["DMC"] = df["DMC"].str[:-2]

    df.to_csv("data/final/log.csv")

    return df



def join_final_val(data, val):

    #val = pd.read_csv("data/final/log.csv").drop(columns=["Unnamed: 0"], errors='ignore')
    #print(val.head())

    #data = pd.read_csv("data/final/filtered_to_rel_Ordn_MAX.csv").drop(columns=["Unnamed: 0"], errors='ignore')
    data = data.groupby(['DMC']).mean().reset_index()
    print(data.head())

    df = pd.merge(how='inner',
                  left=data,
                  right=val,
                  left_on='DMC',
                  right_on='DMC')

    print(df.head())

    df.to_csv("data/final/verficate_MAX.csv")

    return df


def create_excel_for_bastian():
    # specify all data paths of hri_data

    file_paths = ["data/hri/fft_rel_9000.csv",
                  "data/hri/all_files_600.csv",
                  #"data/hri/for_development_11_07_1000_neu.csv"    # can´t process, too much data for RAM
                  ]

    dataframes = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        df = df.groupby(['DMC']).mean()
        df = df.reset_index(level=0)
        df = df.drop(columns=["Unnamed: 0",
                              "id",
                              "Alarmlevel",
                              "TeilNr"],
                     errors='ignore')
        df = df.iloc[:, :7]  # cut away columns for reduced memory
        # print(df.columns)
        dataframes.append(df)
        del df
    df_data = pd.concat(dataframes)
    del dataframes, file_paths

    df_noti_ids = pd.read_excel('data/hri/hri_ord_19_8_max_002.xlsx', engine='openpyxl')    # read in Bastians_excel
    df_noti_ids.rename(columns={"part_id": "DMC"}, inplace=True)


    # concat data and bastians excel
    df_all = pd.concat([df_data, df_noti_ids])
    del df_data, df_noti_ids

    # safe it as csv
    df_all.to_csv("data/final/hri_data_ids_for_bastian.csv")

    #############################################################################################


def transform_merge():

    """
    !!!! Need to execute get_hri_fft_data_from_s3() & get_eol_data() before !!!!
    :return:
    """

    # specify all data paths of hri_data
    hri_files = []
    hri = pd.read_csv("data/hri/fft_rel_9000.csv")  # read in the hri-data, which have been processed in the step before
    hri_600_02 = pd.read_csv("data/hri/all_files_600.csv")
    hri_1000_02 = pd.read_csv("data/hri/for_development_11_07_1000_neu.csv")
    hri_noticable = pd.read_excel('data/hri/hri_ord_19_8_max_002.xlsx',
                                 engine='openpyxl')

    for file in hri_files:
        pass

    # process first file
    hri = hri.groupby(['DMC']).mean()       # aggregate data
    hri = hri.reset_index(level=0)          # reset the index, so that it counts correctly again, after aggregating
    hri = hri.drop(columns=["Unnamed: 0",   # drop not needed columns from the hri data
                            "increment",
                            "id",
                            "processtep",
                            "TimeStamp",
                            "Alarmlevel",
                            "TeilNr",
                            "Drehzahl"])

    # process second file
    hri_600_02 = hri_600_02.groupby(['DMC']).mean()
    hri_600_02 = hri_600_02.reset_index(level=0)

    # proces third file
    hri_1000_02 = hri_1000_02.groupby(['DMC']).mean()
    hri_1000_02 = hri_1000_02.reset_index(level=0)

    # process fourth file - excel


    print(hri.head())
    print(hri_600_02.head())
    print(hri_1000_02.head())
    print(hri_noticable.head())


    hri.to_csv("data/hri/fft_data_grouped.csv")  # safe a copy in the meantime for safety, to come back to
    # DMC; Wert1; ...; Wert850

    ########################################################################################

    relevant_ids = get_noticable_ids_from_excel()

    merged = pd.merge(how='left',
                      left=relevant_ids,
                      right=hri,
                      left_on='part_id',
                      right_on='DMC')\
                      .drop(columns=["DMC"])


    ########################################################################################
    #### Do we need this part?
    #### get_hri_fft_data_from_s3() already filters for the relevant ids !!!!
    ########################################################################################


    eol, eol_values, eol_axis = get_eol_data()  # get the eol data


    result = pd.merge(how='left',   # the merged data is pre filtered for the right id´s !
                      left=merged,
                      right=eol_values,
                      left_on='engine_id',
                      right_on='part_id')

    # drop rows, where values are missing
    nan_value = float("NaN")
    result.replace("",
                   nan_value,
                   inplace=True)
    result.dropna(subset=["Wert1", "test_protocol_id"],
                  inplace=True)

    result.to_csv("data/processed/merged_df_ids_hri_fft_eol_drs_drz.csv")   # make a safety copy


    ######################################################################################


    df = result #TODO: to be deleted

    # get the information from the eol_axis
    eol_axis = eol_axis["curve_values"].apply(json.loads)
    first_axis = eol_axis.iloc[0]

    # relevant meta-data dataframe
    meta = df.loc[:, ['part_id_x', 'engine_id', 'Date']]
    meta = meta.drop_duplicates(subset=['engine_id'],
                                keep='first')

    # make curve values column to list
    df["curve_values"] = df["curve_values"].apply(json.loads)
    df = df.drop_duplicates(subset=['engine_id'],   # drop dublicates, NEED to specify further in future #TODO
                            keep='first')

    # extract the HRI-FFT data
    hri_fft_input_df = df.iloc[:, 6:856]  # Wert1 - Wert850
    hri_19_8_input_df = df.iloc[:, 2]     # extract only Ordn. 19.8
    hri_19_8_input_df.rename('hri_19.8')


    # extract the EOL data
    eol_output_df = df["curve_values"].apply(pd.Series)
    # new_names = [f"curve_value{x}" for x in range(len(df["curve_values"][0]))]
    new_names = [f"eol_ordn_{x}" for x in first_axis]
    old_names = eol_output_df.columns
    eol_output_df.rename(columns=dict(zip(str(old_names), first_axis)),
                         inplace=True)

    df = pd.concat([meta, hri_19_8_input_df, eol_output_df], axis=1)    # put everything together
    df.to_csv("data/processed/meta_idk_idk.csv")

    print(df)

    Ordnung_as_index = int(20 / 0.25)  # 80
    eol_20_df = eol_output_df.iloc[:, Ordnung_as_index]
    eol_20_df.name = 'eol_20'

    # take log of the hri data, like Jakob Bonart
    hri_20_log = np.log(hri_19_8_input_df)

    # now concat what we need into dataframe
    multi_dot_plot_df = pd.concat([meta, hri_20_log, eol_20_df], axis=1)
    print("multi_dot_plot_df: \n", multi_dot_plot_df.head())
    multi_dot_plot_df.to_csv("data/final/meta_hri198_eol20_pre.csv")
    # part_id_x
    # engine_id
    # Date
    # Amplitude_19_8_gr_100
    # eol_20



    ######################################################################################
    ######################################################################################
    #### VISUALIZATION ###################################################################
    ######################################################################################
    ######################################################################################


    vis = False

    if vis:

        corr = df.corr(method='pearson').iloc[0, :]
        corr = pd.DataFrame(corr)
        corr = corr.reset_index()
        corr.columns = ['categories', 'correlation']

        corr_min = int(10 / 0.25)
        corr_max = int(30 / 0.25)
        print("diagram data: ", corr[corr_min:corr_max])


        ax = sns.lineplot(x='categories',
                          y='correlation',
                          data=corr[corr_min:corr_max])
        ax.set_title("correlation plot")
        ax.set_ylabel("pearson-correlation")
        ax.set_xlabel("Ordnungen")
        plt.show()

        # print the relevant dataframes for refereence
        print("correlation: \n", corr.head())
        print("hri_19_8_input_df: \n", hri_19_8_input_df.head())
        print("eol_output_df: \n", eol_output_df.head())

        # dive deeper into analysis and plot the max_corr_pegel (x-axis) and log(HRI) (y-axis)
        # first filter down the EOL-data to 20.Ordnung
        # 4, 55448E+15
        # 4, 34206E+15
        # 4, 32117E+15
        # 4, 04735E+15
        Ordnung_as_index = int(20 / 0.25)  # 80
        eol_20_df = eol_output_df.iloc[:, Ordnung_as_index]
        eol_20_df.name = 'eol_20'

        # take log of the hri data, lake Jakob Bonart
        hri_20_log = np.log(hri_19_8_input_df)

        # now concat what we need into dataframe
        multi_dot_plot_df = pd.concat([meta, hri_20_log, eol_20_df], axis=1)
        print("multi_dot_plot_df: \n", multi_dot_plot_df.head())
        multi_dot_plot_df.to_csv("data/final/multi_dot_plot_df.csv")        ###TODO: ##TODO:

        if True:
            ax = sns.regplot(x='eol_20',
                             y='Amplitude_19_8_gr_100',
                             data=multi_dot_plot_df)
            ax.set_title("Pegel an max-corr")
            ax.set_ylabel("HRI - 19.8")
            ax.set_xlabel("EOL - 20.0")
            plt.show()

        # plt.show()


    ######################################################################################
    ######################################################################################
    ######################################################################################


    hri_eol_funk = False

    if hri_eol_funk:

        # convert dataframes to numpy matrix
        hri_fft_input = hri_fft_input_df.to_numpy()  # (15, 850)
        hri_19_8_input = hri_19_8_input_df.to_numpy().reshape(-1, 1)  # (15, 1)
        eol_output = eol_output_df.to_numpy()  # (15, 1024)

        def correlation_plot(hri_fft_input, hri_19_8_input, eol_output):
            # seaborn is working with pandas dataframes, now with numpy arrays
            # show some insights into the data
            corr_plot_df = pd.DataFrame(np.average(hri_19_8_input,
                                                   axis=1),
                                        columns=["hri_19_8_avg"])
            corr_plot_df["eol_avg"] = pd.DataFrame(np.average(eol_output,
                                                              axis=1))

            # ax = sns.regplot(x="hri_19_8_avg", y="eol_avg", data=corr_plot_df)
            #
            # plt.figure(figsize=(14, 8))
            # sns.set_theme(style="white")
            # corr = corr_plot_df.corr()
            # heatmap = sns.heatmap(corr, annot=True, cmap="Blues", fmt='.1g')

            # try to rebuild the correlation-plots
            corr_plot_df = corr_plot_df.to_numpy()
            rho = np.corrcoef(corr_plot_df)
            print(corr_plot_df)
            print(rho)

            quit()

            # fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 3))
            # for i in [0, 1, 2]:
            #     ax[i].scatter(x[0,], x[1 + i,])
            #     ax[i].title.set_text('Correlation = ' + "{:.2f}".format(rho[0, i + 1]))
            #     ax[i].set(xlabel='x', ylabel='y')
            # fig.subplots_adjust(wspace=.4)
            # plt.show()

            plt.show()

        def plot_spek(hri_input, eol_output):
            # hri: 850 values
            # eol: 1043 values

            x_axis = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25,
                      4.5, 4.75, 5.0, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5, 6.75, 7.0, 7.25, 7.5, 7.75, 8.0, 8.25, 8.5, 8.75,
                      9.0, 9.25, 9.5, 9.75, 10.0, 10.25, 10.5, 10.75, 11.0, 11.25, 11.5, 11.75, 12.0, 12.25, 12.5,
                      12.75, 13.0, 13.25, 13.5, 13.75, 14.0, 14.25, 14.5, 14.75, 15.0, 15.25, 15.5, 15.75, 16.0, 16.25,
                      16.5, 16.75, 17.0, 17.25, 17.5, 17.75, 18.0, 18.25, 18.5, 18.75, 19.0, 19.25, 19.5, 19.75, 20.0,
                      20.25, 20.5, 20.75, 21.0, 21.25, 21.5, 21.75, 22.0, 22.25, 22.5, 22.75, 23.0, 23.25, 23.5, 23.75,
                      24.0, 24.25, 24.5, 24.75, 25.0, 25.25, 25.5, 25.75, 26.0, 26.25, 26.5, 26.75, 27.0, 27.25, 27.5,
                      27.75, 28.0, 28.25, 28.5, 28.75, 29.0, 29.25, 29.5, 29.75, 30.0, 30.25, 30.5, 30.75, 31.0, 31.25,
                      31.5, 31.75, 32.0, 32.25, 32.5, 32.75, 33.0, 33.25, 33.5, 33.75, 34.0, 34.25, 34.5, 34.75, 35.0,
                      35.25, 35.5, 35.75, 36.0, 36.25, 36.5, 36.75, 37.0, 37.25, 37.5, 37.75, 38.0, 38.25, 38.5, 38.75,
                      39.0, 39.25, 39.5, 39.75, 40.0, 40.25, 40.5, 40.75, 41.0, 41.25, 41.5, 41.75, 42.0, 42.25, 42.5,
                      42.75, 43.0, 43.25, 43.5, 43.75, 44.0, 44.25, 44.5, 44.75, 45.0, 45.25, 45.5, 45.75, 46.0, 46.25,
                      46.5, 46.75, 47.0, 47.25, 47.5, 47.75, 48.0, 48.25, 48.5, 48.75, 49.0, 49.25, 49.5, 49.75, 50.0,
                      50.25, 50.5, 50.75, 51.0, 51.25, 51.5, 51.75, 52.0, 52.25, 52.5, 52.75, 53.0, 53.25, 53.5, 53.75,
                      54.0, 54.25, 54.5, 54.75, 55.0, 55.25, 55.5, 55.75, 56.0, 56.25, 56.5, 56.75, 57.0, 57.25, 57.5,
                      57.75, 58.0, 58.25, 58.5, 58.75, 59.0, 59.25, 59.5, 59.75, 60.0, 60.25, 60.5, 60.75, 61.0, 61.25,
                      61.5, 61.75, 62.0, 62.25, 62.5, 62.75, 63.0, 63.25, 63.5, 63.75, 64.0, 64.25, 64.5, 64.75, 65.0,
                      65.25, 65.5, 65.75, 66.0, 66.25, 66.5, 66.75, 67.0, 67.25, 67.5, 67.75, 68.0, 68.25, 68.5, 68.75,
                      69.0, 69.25, 69.5, 69.75, 70.0, 70.25, 70.5, 70.75, 71.0, 71.25, 71.5, 71.75, 72.0, 72.25, 72.5,
                      72.75, 73.0, 73.25, 73.5, 73.75, 74.0, 74.25, 74.5, 74.75, 75.0, 75.25, 75.5, 75.75, 76.0, 76.25,
                      76.5, 76.75, 77.0, 77.25, 77.5, 77.75, 78.0, 78.25, 78.5, 78.75, 79.0, 79.25, 79.5, 79.75, 80.0,
                      80.25, 80.5, 80.75, 81.0, 81.25, 81.5, 81.75, 82.0, 82.25, 82.5, 82.75, 83.0, 83.25, 83.5, 83.75,
                      84.0, 84.25, 84.5, 84.75, 85.0, 85.25, 85.5, 85.75, 86.0, 86.25, 86.5, 86.75, 87.0, 87.25, 87.5,
                      87.75, 88.0, 88.25, 88.5, 88.75, 89.0, 89.25, 89.5, 89.75, 90.0, 90.25, 90.5, 90.75, 91.0, 91.25,
                      91.5, 91.75, 92.0, 92.25, 92.5, 92.75, 93.0, 93.25, 93.5, 93.75, 94.0, 94.25, 94.5, 94.75, 95.0,
                      95.25, 95.5, 95.75, 96.0, 96.25, 96.5, 96.75, 97.0, 97.25, 97.5, 97.75, 98.0, 98.25, 98.5, 98.75,
                      99.0, 99.25, 99.5, 99.75, 100.0, 100.25, 100.5, 100.75, 101.0, 101.25, 101.5, 101.75, 102.0,
                      102.25, 102.5, 102.75, 103.0, 103.25, 103.5, 103.75, 104.0, 104.25, 104.5, 104.75, 105.0, 105.25,
                      105.5, 105.75, 106.0, 106.25, 106.5, 106.75, 107.0, 107.25, 107.5, 107.75, 108.0, 108.25, 108.5,
                      108.75, 109.0, 109.25, 109.5, 109.75, 110.0, 110.25, 110.5, 110.75, 111.0, 111.25, 111.5, 111.75,
                      112.0, 112.25, 112.5, 112.75, 113.0, 113.25, 113.5, 113.75, 114.0, 114.25, 114.5, 114.75, 115.0,
                      115.25, 115.5, 115.75, 116.0, 116.25, 116.5, 116.75, 117.0, 117.25, 117.5, 117.75, 118.0, 118.25,
                      118.5, 118.75, 119.0, 119.25, 119.5, 119.75, 120.0, 120.25, 120.5, 120.75, 121.0, 121.25, 121.5,
                      121.75, 122.0, 122.25, 122.5, 122.75, 123.0, 123.25, 123.5, 123.75, 124.0, 124.25, 124.5, 124.75,
                      125.0, 125.25, 125.5, 125.75, 126.0, 126.25, 126.5, 126.75, 127.0, 127.25, 127.5, 127.75, 128.0,
                      128.25, 128.5, 128.75, 129.0, 129.25, 129.5, 129.75, 130.0, 130.25, 130.5, 130.75, 131.0, 131.25,
                      131.5, 131.75, 132.0, 132.25, 132.5, 132.75, 133.0, 133.25, 133.5, 133.75, 134.0, 134.25, 134.5,
                      134.75, 135.0, 135.25, 135.5, 135.75, 136.0, 136.25, 136.5, 136.75, 137.0, 137.25, 137.5, 137.75,
                      138.0, 138.25, 138.5, 138.75, 139.0, 139.25, 139.5, 139.75, 140.0, 140.25, 140.5, 140.75, 141.0,
                      141.25, 141.5, 141.75, 142.0, 142.25, 142.5, 142.75, 143.0, 143.25, 143.5, 143.75, 144.0, 144.25,
                      144.5, 144.75, 145.0, 145.25, 145.5, 145.75, 146.0, 146.25, 146.5, 146.75, 147.0, 147.25, 147.5,
                      147.75, 148.0, 148.25, 148.5, 148.75, 149.0, 149.25, 149.5, 149.75, 150.0, 150.25, 150.5, 150.75,
                      151.0, 151.25, 151.5, 151.75, 152.0, 152.25, 152.5, 152.75, 153.0, 153.25, 153.5, 153.75, 154.0,
                      154.25, 154.5, 154.75, 155.0, 155.25, 155.5, 155.75, 156.0, 156.25, 156.5, 156.75, 157.0, 157.25,
                      157.5, 157.75, 158.0, 158.25, 158.5, 158.75, 159.0, 159.25, 159.5, 159.75, 160.0, 160.25, 160.5,
                      160.75, 161.0, 161.25, 161.5, 161.75, 162.0, 162.25, 162.5, 162.75, 163.0, 163.25, 163.5, 163.75,
                      164.0, 164.25, 164.5, 164.75, 165.0, 165.25, 165.5, 165.75, 166.0, 166.25, 166.5, 166.75, 167.0,
                      167.25, 167.5, 167.75, 168.0, 168.25, 168.5, 168.75, 169.0, 169.25, 169.5, 169.75, 170.0, 170.25,
                      170.5, 170.75, 171.0, 171.25, 171.5, 171.75, 172.0, 172.25, 172.5, 172.75, 173.0, 173.25, 173.5,
                      173.75, 174.0, 174.25, 174.5, 174.75, 175.0, 175.25, 175.5, 175.75, 176.0, 176.25, 176.5, 176.75,
                      177.0, 177.25, 177.5, 177.75, 178.0, 178.25, 178.5, 178.75, 179.0, 179.25, 179.5, 179.75, 180.0,
                      180.25, 180.5, 180.75, 181.0, 181.25, 181.5, 181.75, 182.0, 182.25, 182.5, 182.75, 183.0, 183.25,
                      183.5, 183.75, 184.0, 184.25, 184.5, 184.75, 185.0, 185.25, 185.5, 185.75, 186.0, 186.25, 186.5,
                      186.75, 187.0, 187.25, 187.5, 187.75, 188.0, 188.25, 188.5, 188.75, 189.0, 189.25, 189.5, 189.75,
                      190.0, 190.25, 190.5, 190.75, 191.0, 191.25, 191.5, 191.75, 192.0, 192.25, 192.5, 192.75, 193.0,
                      193.25, 193.5, 193.75, 194.0, 194.25, 194.5, 194.75, 195.0, 195.25, 195.5, 195.75, 196.0, 196.25,
                      196.5, 196.75, 197.0, 197.25, 197.5, 197.75, 198.0, 198.25, 198.5, 198.75, 199.0, 199.25, 199.5,
                      199.75, 200.0, 200.25, 200.5, 200.75, 201.0, 201.25, 201.5, 201.75, 202.0, 202.25, 202.5, 202.75,
                      203.0, 203.25, 203.5, 203.75, 204.0, 204.25, 204.5, 204.75, 205.0, 205.25, 205.5, 205.75, 206.0,
                      206.25, 206.5, 206.75, 207.0, 207.25, 207.5, 207.75, 208.0, 208.25, 208.5, 208.75, 209.0, 209.25,
                      209.5, 209.75, 210.0, 210.25, 210.5, 210.75, 211.0, 211.25, 211.5, 211.75, 212.0, 212.25, 212.5,
                      212.75, 213.0, 213.25, 213.5, 213.75, 214.0, 214.25, 214.5, 214.75, 215.0, 215.25, 215.5, 215.75,
                      216.0, 216.25, 216.5, 216.75, 217.0, 217.25, 217.5, 217.75, 218.0, 218.25, 218.5, 218.75, 219.0,
                      219.25, 219.5, 219.75, 220.0, 220.25, 220.5, 220.75, 221.0, 221.25, 221.5, 221.75, 222.0, 222.25,
                      222.5, 222.75, 223.0, 223.25, 223.5, 223.75, 224.0, 224.25, 224.5, 224.75, 225.0, 225.25, 225.5,
                      225.75, 226.0, 226.25, 226.5, 226.75, 227.0, 227.25, 227.5, 227.75, 228.0, 228.25, 228.5, 228.75,
                      229.0, 229.25, 229.5, 229.75, 230.0, 230.25, 230.5, 230.75, 231.0, 231.25, 231.5, 231.75, 232.0,
                      232.25, 232.5, 232.75, 233.0, 233.25, 233.5, 233.75, 234.0, 234.25, 234.5, 234.75, 235.0, 235.25,
                      235.5, 235.75, 236.0, 236.25, 236.5, 236.75, 237.0, 237.25, 237.5, 237.75, 238.0, 238.25, 238.5,
                      238.75, 239.0, 239.25, 239.5, 239.75, 240.0, 240.25, 240.5, 240.75, 241.0, 241.25, 241.5, 241.75,
                      242.0, 242.25, 242.5, 242.75, 243.0, 243.25, 243.5, 243.75, 244.0, 244.25, 244.5, 244.75, 245.0,
                      245.25, 245.5, 245.75, 246.0, 246.25, 246.5, 246.75, 247.0, 247.25, 247.5, 247.75, 248.0, 248.25,
                      248.5, 248.75, 249.0, 249.25, 249.5, 249.75, 250.0, 250.25, 250.5, 250.75, 251.0, 251.25, 251.5,
                      251.75, 252.0, 252.25, 252.5, 252.75, 253.0, 253.25, 253.5, 253.75, 254.0, 254.25, 254.5, 254.75,
                      255.0, 255.25, 255.5, 255.75]

            count_rows = eol_output.shape[0]

            colors = []
            for name, hex in matplotlib.colors.cnames.items():
                colors.append(name)
            random.shuffle(colors)

            fig, ax = plt.subplots(count_rows - 7,
                                   4,
                                   # figsize=(8, 9),
                                   constrained_layout=True)

            for i in range(count_rows):
                if i < 8:
                    ax[i, 0].plot(x_axis[:eol_output.shape[1]],
                                  eol_output[i, :],
                                  linewidth=0.4,
                                  linestyle="-",
                                  color='darkolivegreen'
                                  )
                    ax[i, 2].plot(x_axis[:hri_input.shape[1]],
                                  hri_input[i, :],
                                  linewidth=0.4,
                                  linestyle="-",
                                  color='darkolivegreen'
                                  )
                else:
                    ax[i - 8, 1].plot(x_axis[:eol_output.shape[1]],
                                      eol_output[i, :],
                                      linewidth=0.4,
                                      linestyle="-",
                                      color='darkolivegreen'
                                      )
                    ax[i - 8, 3].plot(x_axis[:hri_input.shape[1]],
                                      hri_input[i, :],
                                      linewidth=0.4,
                                      linestyle="-",
                                      color='darkolivegreen'
                                      )

            plt.show()

        def create_spectrum_eol(data):
            """
            :param data: data you want to see a spektrum of
            :return: no return, just display the data as spektrum
            """
            path = '../data/eol/Query_results_1.csv'

            df = pd.read_csv(path)

            df0 = df[df.part_id == 'H894B220']
            print(df0)

            fig, ax = plt.subplots(3, 1, figsize=(8, 9))
            ax[0].plot(json.loads(df0.curve_values[3])[:],
                       json.loads(df0.curve_values[0])[:],
                       linewidth=0.5,
                       linestyle="-",
                       color="mediumblue"
                       )
            ax[1].plot(json.loads(df0.curve_values[3])[:],
                       json.loads(df0.curve_values[1])[:],
                       linewidth=0.5,
                       linestyle="-",
                       color="purple"
                       )
            ax[2].plot(json.loads(df0.curve_values[3])[:],
                       json.loads(df0.curve_values[2])[:],
                       linewidth=0.5,
                       linestyle="-",
                       color="orange"
                       )
            ax[0].set_title(df0.variable[0])
            ax[1].set_title(df0.variable[1])
            ax[2].set_title(df0.variable[2])
            plt.show()

            print(df0.curve_values[3])
            ab = json.loads(df0.curve_values[3])
            print(ab)
            print(ab[:30])

            print(len(df0.curve_values[3]))


def main():

    ### TRIGGER TO GET DATA FROM S3 - default not do it
    if False:
        get_hri_fft_data_from_s3()

    #### process the downloaded files
    #TODO: !!!! if you read in other files, change hardcoded path in stack_hri()-method
    df = stack_hri()    # walk all processed files along and stack them underneath
    df = next_operation(df)
    df = get_eol_process_and_join(df)
    df = filter_to_needed(df)

    val = validierung_hri_data()

    df = join_final_val(df, val)


if __name__ == '__main__':
    main()

